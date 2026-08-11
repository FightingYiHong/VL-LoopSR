#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run CPU-friendly open-source symbolic-regression baselines on the local
benchmark wrappers.

The first supported method is gplearn because it is installed and smoke-tested
in the `srbench-gplearn` environment. This runner reuses the existing benchmark
selection/loading code from:
  - scripts/run_sldbench.py
  - scripts/run_srds.py
  - scripts/run_SRbench.py
  - scripts/run_llmsrbench.py

It prints in the same per-case style as those wrappers and adds:
  - train/val/test R^2
  - expression complexity/depth/string length
  - perfect_fit_by_r2
"""

import argparse
import contextlib
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_metrics import evaluate_expression
from benchmark_metrics import expression_complexity as sympy_expression_complexity

RUN_TIMESTAMP = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{int((time.time() % 1.0) * 1000):03d}"

DEFAULT_RESULTS_BASE = os.environ.get("CPU_SR_RESULTS_BASE", str(PROJECT_ROOT / "runs" / "cpu_baselines"))
MSE_THRESHOLD = float(os.environ.get("CPU_SR_MSE_THRESHOLD", "100.0"))
PERFECT_FIT_TOL = float(os.environ.get("CPU_SR_PERFECT_FIT_TOL", "1e-10"))
PERFECT_FIT_R2_THRESHOLD = float(os.environ.get("CPU_SR_PERFECT_FIT_R2_THRESHOLD", "0.999"))
CASE_LOG_TAIL_CHARS = int(os.environ.get("CPU_SR_CASE_LOG_TAIL_CHARS", "4000"))
SEARCH_EFFICIENCY_THRESHOLD = float(os.environ.get("CPU_SR_SEARCH_EFFICIENCY_THRESHOLD", "0.999"))


class ValidationSearchHit(RuntimeError):
    """Internal control flow used to stop a search after the validation hit."""


class SearchEfficiencyTracker:
    def __init__(self, method: str):
        self.method = method
        self.threshold = SEARCH_EFFICIENCY_THRESHOLD
        self.started_at = time.time()
        self.observed_evaluations = 0
        self.unique_evaluations = 0
        self.native_evaluations_observed = 0
        self.seen = set()
        self.first_hit = None
        self.best = None
        trace_path = os.environ.get("CPU_SR_SEARCH_EFFICIENCY_TRACE")
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            if self.trace_path.exists():
                self.trace_path.unlink()
        self._trace_fp = None

    @staticmethod
    def _key(expr):
        return re.sub(r"\s+", "", str(expr))

    def observe(self, expr, pred_val, *, stage: str, native_evaluations=None, extra=None):
        self.observed_evaluations += 1
        if native_evaluations is not None:
            self.native_evaluations_observed = max(
                self.native_evaluations_observed,
                int(native_evaluations),
            )
        key = self._key(expr)
        is_unique = key not in self.seen
        if is_unique:
            self.seen.add(key)
            self.unique_evaluations += 1
        val_r2 = safe_r2(pred_val[0], pred_val[1])
        event = {
            "method": self.method,
            "candidate_index": int(self.observed_evaluations),
            "unique_candidate_index": int(self.unique_evaluations),
            "is_unique": bool(is_unique),
            "native_evaluations": (
                int(native_evaluations) if native_evaluations is not None else None
            ),
            "stage": str(stage),
            "expression": str(expr),
            "validation_r2": val_r2,
            "elapsed_sec": float(time.time() - self.started_at),
        }
        if extra:
            event.update(json_safe_record(extra))
        if self.best is None or (
            val_r2 is not None and val_r2 > (self.best.get("validation_r2") or -math.inf)
        ):
            self.best = dict(event)
        if self.first_hit is None and val_r2 is not None and val_r2 > self.threshold:
            event["validation_success"] = True
            self.first_hit = dict(event)
        else:
            event["validation_success"] = False
        self._write_event(event)
        return bool(event["validation_success"])

    def _write_event(self, event):
        if not self.trace_path:
            return
        if (
            not event.get("is_unique")
            and not event.get("validation_success")
            and int(event["candidate_index"]) % 1000 != 0
        ):
            return
        if self._trace_fp is None:
            opener = gzip.open if self.trace_path.suffix == ".gz" else open
            self._trace_fp = opener(self.trace_path, "wt", encoding="utf-8")
        self._trace_fp.write(
            json.dumps(json_safe_record(event), ensure_ascii=False, allow_nan=False)
        )
        self._trace_fp.write("\n")
        self._trace_fp.flush()

    def metrics(self):
        if self._trace_fp is not None:
            self._trace_fp.close()
            self._trace_fp = None
        hit = self.first_hit or {}
        best = self.best or {}
        elapsed = max(time.time() - self.started_at, 1e-12)
        return {
            "validation_search_threshold": float(self.threshold),
            "validation_search_rule": "strictly_greater",
            "validation_search_success": bool(self.first_hit),
            "evaluations_to_validation_success": hit.get("candidate_index"),
            "unique_evaluations_to_validation_success": hit.get("unique_candidate_index"),
            "native_evaluations_to_validation_success": hit.get("native_evaluations"),
            "first_validation_success_stage": hit.get("stage"),
            "first_validation_success_expression": hit.get("expression"),
            "first_validation_success_r2": hit.get("validation_r2"),
            "first_validation_success_elapsed_sec": hit.get("elapsed_sec"),
            "validation_search_observed_evaluations": int(self.observed_evaluations),
            "validation_search_unique_evaluations": int(self.unique_evaluations),
            "validation_search_native_evaluations": int(self.native_evaluations_observed),
            "validation_search_best_r2": best.get("validation_r2"),
            "validation_search_best_expression": best.get("expression"),
            "validation_search_evaluations_per_sec": float(
                self.observed_evaluations / elapsed
            ),
            "validation_search_trace": str(self.trace_path) if self.trace_path else None,
        }


def _predict_sympy_candidate(expr, X, variable_names):
    import sympy as sp

    symbols = sp.symbols(" ".join(variable_names))
    if len(variable_names) == 1:
        symbols = (symbols,)
    locals_map = {
        **{str(sym): sym for sym in symbols},
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "sign": sp.sign,
    }
    parsed = sp.sympify(str(expr).replace("^", "**"), locals=locals_map)
    fn = sp.lambdify(symbols, parsed, modules=["numpy"])
    pred = np.asarray(fn(*[X[:, i] for i in range(X.shape[1])]), dtype=float)
    if pred.shape == ():
        pred = np.full(X.shape[0], float(pred), dtype=float)
    pred = pred.reshape(-1)
    if len(pred) != len(X) or not np.all(np.isfinite(pred)):
        raise ValueError("candidate produced invalid validation predictions")
    return pred

def sanitize_name(text: str) -> str:
    text = str(text).replace("/", "__").replace("\\", "__")
    text = re.sub(r"[^a-zA-Z0-9_.\-]+", "_", text)
    suffix = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{text}_{suffix}"


def import_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dataframe_to_xy(df: pd.DataFrame):
    if "y" not in df.columns:
        raise KeyError("expected a target column named 'y'")
    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return X, y, feature_cols


def safe_mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return None
    return float(np.mean((y_true - y_pred) ** 2))


def safe_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return None
    return float(np.mean(np.abs(y_true - y_pred)))


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return 1.0 if ss_res <= PERFECT_FIT_TOL else 0.0
    return float(1.0 - ss_res / ss_tot)


def update_function_list(function_set):
    if function_set is None:
        return None
    from gplearn.functions import make_function

    extra_func_dict = {
        "exp": make_function(function=np.exp, name="exp", arity=1),
        "pow": make_function(function=np.power, name="pow", arity=2),
    }
    return [extra_func_dict.get(func_str, func_str) for func_str in function_set]


def load_gplearn_config(config_path: str):
    with open(os.path.expanduser(config_path), "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    kwargs = dict(config.get("model", {}).get("kwargs", {}))
    kwargs["function_set"] = update_function_list(kwargs.get("function_set"))
    return kwargs


def gplearn2sympy(eq_str):
    import sympy

    str2sympy = {
        "neg": lambda x: -x,
        "abs": sympy.Abs,
        "sqrt": sympy.sqrt,
        "exp": sympy.exp,
        "log": sympy.log,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "pow": lambda x, y: x ** y,
    }
    eq_str_w_lower_vars = re.sub(r"\bX([0-9]*[0-9])\b", r"x\1", str(eq_str))
    return sympy.sympify(eq_str_w_lower_vars, locals=str2sympy)


def expression_complexity(model, expr_str: str):
    import sympy

    program = getattr(model, "_program", None)
    length = getattr(program, "length_", None)
    depth = getattr(program, "depth_", None)
    try:
        sympy_nodes = int(sympy.count_ops(gplearn2sympy(expr_str), visual=False))
    except Exception:
        sympy_nodes = None
    return {
        "expr_complexity": int(length) if length is not None else len(str(expr_str).replace(" ", "")),
        "expr_depth": int(depth) if depth is not None else None,
        "expr_string_length": int(len(str(expr_str).replace(" ", ""))),
        "expr_sympy_ops": sympy_nodes,
    }


def _patch_gplearn_validate_data(model):
    if hasattr(model, "_validate_data"):
        return

    import types
    from sklearn.utils.validation import check_array, check_X_y

    def _validate_data(self, X, y=None, y_numeric=False, **kwargs):
        if y is None:
            X_checked = check_array(X, **kwargs)
            self.n_features_in_ = X_checked.shape[1]
            return X_checked
        X_checked, y_checked = check_X_y(X, y, y_numeric=y_numeric, **kwargs)
        self.n_features_in_ = X_checked.shape[1]
        return X_checked, y_checked

    model._validate_data = types.MethodType(_validate_data, model)


def fit_gplearn(train_df, val_df, test_df, config_path, random_state=None):
    from gplearn.genetic import SymbolicRegressor

    X_train, y_train, feature_cols = dataframe_to_xy(train_df)
    X_val, y_val, _ = dataframe_to_xy(val_df)
    X_test, y_test, _ = dataframe_to_xy(test_df)

    config = load_yaml_config(config_path)
    kwargs = dict(config.get("model", {}).get("kwargs", {}))
    kwargs["function_set"] = update_function_list(kwargs.get("function_set"))
    data_config = dict(config.get("data", {}))
    runtime_config = dict(config.get("runtime", {}))
    if random_state is not None and "random_state" not in kwargs:
        kwargs["random_state"] = int(random_state)
    fit_X_train, fit_y_train, train_subsampled = _sample_xy_rows(
        X_train,
        y_train,
        data_config.get("max_train_rows"),
        random_state=random_state,
    )

    tracker = SearchEfficiencyTracker("gplearn")

    class TracedSymbolicRegressor(SymbolicRegressor):
        def _verbose_reporter(self, run_details=None):
            if run_details is None or not getattr(self, "_programs", None):
                return
            generation = int(run_details["generation"][-1])
            population = self._programs[-1]
            native_generation_end = int((generation + 1) * self.population_size)
            for candidate_index, program in enumerate(population, start=1):
                if program is None:
                    continue
                try:
                    pred = np.asarray(program.execute(X_val), dtype=float).reshape(-1)
                    hit = tracker.observe(
                        str(program),
                        (y_val, pred),
                        stage=f"generation_{generation}",
                        native_evaluations=native_generation_end,
                        extra={
                            "generation": generation,
                            "population_index": candidate_index,
                            "program_length": getattr(program, "length_", None),
                            "program_depth": getattr(program, "depth_", None),
                        },
                    )
                except Exception:
                    continue
                if hit:
                    self._validation_hit_program = program
                    raise ValidationSearchHit

    # gplearn only invokes its generation callback in verbose mode. The
    # subclass suppresses console reporting and uses it for candidate tracing.
    kwargs["verbose"] = 1
    kwargs["low_memory"] = False
    model = TracedSymbolicRegressor(**kwargs)
    _patch_gplearn_validate_data(model)
    fit_start = time.time()
    try:
        with _time_limit(runtime_config.get("max_fit_seconds")):
            model.fit(fit_X_train, fit_y_train)
    except ValidationSearchHit:
        model._program = model._validation_hit_program
    fit_sec = time.time() - fit_start

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
    expr = str(model._program)

    test_mse = safe_mse(y_test, pred_test)
    test_r2 = safe_r2(y_test, pred_test)
    metrics = {
        "method": "gplearn",
        "valid_formula_found": True,
        "num_candidate_exprs": 1,
        "best_expr": expr,
        "best_expr_sympy": str(gplearn2sympy(expr)) if expr else None,
        "best_train_mse": safe_mse(y_train, pred_train),
        "best_val_mse": safe_mse(y_val, pred_val),
        "best_test_mse": test_mse,
        "train_mae": safe_mae(y_train, pred_train),
        "val_mae": safe_mae(y_val, pred_val),
        "test_mae": safe_mae(y_test, pred_test),
        "train_r2": safe_r2(y_train, pred_train),
        "val_r2": safe_r2(y_val, pred_val),
        "test_r2": test_r2,
        "passed": bool(test_mse is not None and test_mse <= MSE_THRESHOLD),
        "perfect_fit": bool(test_mse is not None and test_mse <= PERFECT_FIT_TOL),
        "perfect_fit_by_r2": bool(test_r2 is not None and test_r2 >= PERFECT_FIT_R2_THRESHOLD),
        "fit_runtime_sec": float(fit_sec),
        "feature_names": " | ".join([str(x) for x in feature_cols]),
        "fit_n_train": int(len(fit_X_train)),
        "fit_train_subsampled": bool(train_subsampled),
        "max_fit_seconds": runtime_config.get("max_fit_seconds"),
    }
    metrics.update(expression_complexity(model, expr))
    metrics.update(tracker.metrics())
    metrics["validation_search_count_semantics"] = (
        "population-order candidate checks; native hit count is the end of the "
        "generation because the full generation is generated before validation"
    )
    return metrics


def load_yaml_config(config_path: str):
    with open(os.path.expanduser(config_path), "r", encoding="utf-8") as fp:
        return yaml.load(fp, Loader=yaml.FullLoader) or {}


def _set_default_random_state(kwargs, random_state):
    if random_state is not None and "random_state" not in kwargs:
        kwargs["random_state"] = int(random_state)
    return kwargs


def _coerce_predictions(pred):
    pred = np.asarray(pred, dtype=float)
    if pred.ndim > 1:
        pred = pred.reshape(-1)
    if not np.isfinite(pred).all():
        raise ValueError("non-finite predictions")
    return pred


def _metric_dict(method, expr, pred_train, pred_val, pred_test, y_train, y_val, y_test, feature_cols, fit_sec, num_candidates=None):
    pred_train = _coerce_predictions(pred_train)
    pred_val = _coerce_predictions(pred_val)
    pred_test = _coerce_predictions(pred_test)
    test_mse = safe_mse(y_test, pred_test)
    test_r2 = safe_r2(y_test, pred_test)
    metrics = {
        "method": method,
        "valid_formula_found": bool(expr),
        "num_candidate_exprs": num_candidates,
        "best_expr": str(expr) if expr is not None else None,
        "best_expr_sympy": str(expr) if expr is not None else None,
        "best_train_mse": safe_mse(y_train, pred_train),
        "best_val_mse": safe_mse(y_val, pred_val),
        "best_test_mse": test_mse,
        "train_mae": safe_mae(y_train, pred_train),
        "val_mae": safe_mae(y_val, pred_val),
        "test_mae": safe_mae(y_test, pred_test),
        "train_r2": safe_r2(y_train, pred_train),
        "val_r2": safe_r2(y_val, pred_val),
        "test_r2": test_r2,
        "passed": bool(test_mse is not None and test_mse <= MSE_THRESHOLD),
        "perfect_fit": bool(test_mse is not None and test_mse <= PERFECT_FIT_TOL),
        "perfect_fit_by_r2": bool(test_r2 is not None and test_r2 >= PERFECT_FIT_R2_THRESHOLD),
        "fit_runtime_sec": float(fit_sec),
        "feature_names": " | ".join([str(x) for x in feature_cols]),
    }
    metrics.update(sympy_expression_complexity(expr, feature_cols))
    return metrics


def _instantiate_with_kwargs(cls, kwargs):
    try:
        return cls(**kwargs)
    except TypeError:
        import inspect

        try:
            allowed = set(inspect.signature(cls).parameters)
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
            return cls(**filtered)
        except Exception:
            raise


@contextlib.contextmanager
def _time_limit(seconds):
    if not seconds or float(seconds) <= 0:
        yield
        return

    seconds = float(seconds)
    old_handler = signal.getsignal(signal.SIGALRM)

    def _handler(signum, frame):
        raise TimeoutError(f"fit exceeded {seconds:.1f} seconds")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _sample_xy_rows(X, y, max_rows, random_state=None):
    if not max_rows or int(max_rows) <= 0 or len(X) <= int(max_rows):
        return X, y, False
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=int(max_rows), replace=False)
    idx.sort()
    return X[idx], y[idx], True


def _pysr_best_expr(model):
    try:
        return str(model.sympy())
    except Exception:
        pass
    equations = getattr(model, "equations_", None)
    if equations is not None and len(equations):
        if "sympy_format" in equations:
            return str(equations.iloc[-1]["sympy_format"])
        if "equation" in equations:
            return str(equations.iloc[-1]["equation"])
    return str(model)


def fit_pysr(train_df, val_df, test_df, config_path, random_state=None):
    from pysr import PySRRegressor

    X_train, y_train, feature_cols = dataframe_to_xy(train_df)
    X_val, y_val, _ = dataframe_to_xy(val_df)
    X_test, y_test, _ = dataframe_to_xy(test_df)
    config = load_yaml_config(config_path)
    data_config = dict(config.get("data", {}))
    fit_X_train, fit_y_train, train_subsampled = _sample_xy_rows(
        X_train,
        y_train,
        data_config.get("max_train_rows"),
        random_state=random_state,
    )
    kwargs = dict(config.get("fit", config.get("model", {}).get("kwargs", {})))
    _set_default_random_state(kwargs, random_state)
    target_iterations = int(kwargs.pop("niterations", 100))
    case_timeout = kwargs.pop("timeout_in_seconds", None)
    # PySR validates names against Julia functions (for example, ``N`` and
    # ``Si`` are reserved).  Stable synthetic names avoid dataset-dependent
    # startup failures while preserving the feature order used for scoring.
    variable_names = [f"x{i}" for i in range(X_train.shape[1])]
    # Keep the timeout inside the Julia search as well as around the Python
    # call.  A Python signal cannot reliably interrupt a long native Julia
    # call, which previously let the outer supervisor kill the process before
    # PySR could return and serialize its Hall-of-Fame.
    if case_timeout is not None:
        kwargs["timeout_in_seconds"] = float(case_timeout)
    ncycles = int(
        kwargs.get("ncycles_per_iteration", kwargs.get("ncyclesperiteration", 380))
    )
    populations = int(kwargs.get("populations", 31))
    population_size = int(kwargs.get("population_size", 27))
    nominal_evals_per_iteration = int(
        populations * math.ceil(population_size / 10.0) * ncycles
    )
    kwargs["niterations"] = 1
    kwargs["warm_start"] = True
    tracker = SearchEfficiencyTracker("pysr")
    model = PySRRegressor(**kwargs)
    fit_start = time.time()
    completed_iterations = 0
    fit_timeout_recovered = False
    try:
        with _time_limit(case_timeout):
            for iteration in range(1, target_iterations + 1):
                try:
                    model.fit(
                        fit_X_train,
                        fit_y_train,
                        variable_names=variable_names,
                    )
                except TypeError:
                    model.fit(fit_X_train, fit_y_train)
                completed_iterations = iteration
                equations = getattr(model, "equations_", None)
                if equations is None:
                    continue
                native_evaluations = int(iteration * nominal_evals_per_iteration)
                for rank, (_, row) in enumerate(equations.iterrows(), start=1):
                    expr = row.get("sympy_format", row.get("equation"))
                    try:
                        pred = _predict_sympy_candidate(
                            expr,
                            X_val,
                            variable_names,
                        )
                        hit = tracker.observe(
                            expr,
                            (y_val, pred),
                            stage=f"iteration_{iteration}_hall_of_fame",
                            native_evaluations=native_evaluations,
                            extra={
                                "iteration": iteration,
                                "hall_of_fame_rank": rank,
                                "complexity": row.get("complexity"),
                                "training_loss": row.get("loss"),
                            },
                        )
                    except Exception:
                        continue
                    if hit:
                        break
                if tracker.first_hit:
                    break
                if case_timeout is not None and time.time() - fit_start >= float(case_timeout):
                    break
    except TimeoutError:
        # PySR writes its Hall-of-Fame during the search.  Reaching the
        # benchmark time budget should freeze and score that final available
        # output, not discard the entire run as if no expression existed.
        equations = getattr(model, "equations_", None)
        if equations is None or len(equations) == 0:
            raise
        fit_timeout_recovered = True
    fit_sec = time.time() - fit_start
    expr = (
        tracker.first_hit.get("expression")
        if tracker.first_hit
        else _pysr_best_expr(model)
    )
    equations = getattr(model, "equations_", None)
    num_candidates = len(equations) if equations is not None else None
    pred_train = _predict_sympy_candidate(expr, X_train, variable_names)
    pred_val = _predict_sympy_candidate(expr, X_val, variable_names)
    pred_test = _predict_sympy_candidate(expr, X_test, variable_names)
    metrics = _metric_dict(
        "pysr",
        expr,
        pred_train,
        pred_val,
        pred_test,
        y_train,
        y_val,
        y_test,
        feature_cols,
        fit_sec,
        num_candidates=num_candidates,
    )
    metrics.update(tracker.metrics())
    metrics.update({
        "fit_n_train": int(len(fit_X_train)),
        "fit_train_subsampled": bool(train_subsampled),
        "pysr_completed_iterations": int(completed_iterations),
        "pysr_fit_timeout_recovered": bool(fit_timeout_recovered),
        "pysr_nominal_evaluations_per_iteration": int(nominal_evals_per_iteration),
        "validation_search_count_semantics": (
            "Hall-of-Fame candidates checked after each PySR iteration; native "
            "evaluation count is the documented iteration-level nominal upper bound"
        ),
    })
    return metrics


def _ffx_best_expr(model):
    ffx_model = getattr(model, "model_", None)
    if ffx_model is not None:
        for attr in ["str2", "__str__"]:
            obj = getattr(ffx_model, attr, None)
            if callable(obj):
                try:
                    return str(obj()).replace("^", "**")
                except Exception:
                    pass
    return str(model)


def fit_ffx(train_df, val_df, test_df, config_path, random_state=None):
    import scipy
    from ffx import FFXRegressor

    for name in ["isfinite", "isinf", "isnan"]:
        if not hasattr(scipy, name):
            setattr(scipy, name, getattr(np, name))

    X_train, y_train, feature_cols = dataframe_to_xy(train_df)
    X_val, y_val, _ = dataframe_to_xy(val_df)
    X_test, y_test, _ = dataframe_to_xy(test_df)
    config = load_yaml_config(config_path)
    data_config = dict(config.get("data", {}))
    runtime_config = dict(config.get("runtime", {}))
    fit_X_train, fit_y_train, train_subsampled = _sample_xy_rows(
        X_train,
        y_train,
        data_config.get("max_train_rows"),
        random_state=random_state,
    )

    model = FFXRegressor()
    fit_start = time.time()
    with _time_limit(runtime_config.get("max_fit_seconds")):
        model.fit(fit_X_train, fit_y_train)
    fit_sec = time.time() - fit_start
    expr = _ffx_best_expr(model)
    metrics = _metric_dict(
        "ffx",
        expr,
        model.predict(X_train),
        model.predict(X_val),
        model.predict(X_test),
        y_train,
        y_val,
        y_test,
        feature_cols,
        fit_sec,
        num_candidates=None,
    )
    ffx_model = getattr(model, "model_", None)
    complexity_fn = getattr(ffx_model, "complexity", None)
    if callable(complexity_fn):
        try:
            metrics["expr_complexity"] = int(complexity_fn())
        except Exception:
            pass
    metrics.update({
        "fit_n_train": int(len(fit_X_train)),
        "fit_train_subsampled": bool(train_subsampled),
        "max_fit_seconds": runtime_config.get("max_fit_seconds"),
    })
    return metrics


def _bingo_best_expr(model, feature_cols):
    try:
        expr = str(model.get_best_individual())
    except Exception:
        expr = str(model)
    for idx, col in reversed(list(enumerate(feature_cols))):
        expr = expr.replace(f"X_{idx}", str(col))
    return expr.replace(")(", ")*(").replace("^", "**")


def fit_bingo(train_df, val_df, test_df, config_path, random_state=None):
    from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
    from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor

    X_train, y_train, feature_cols = dataframe_to_xy(train_df)
    X_val, y_val, _ = dataframe_to_xy(val_df)
    X_test, y_test, _ = dataframe_to_xy(test_df)
    config = load_yaml_config(config_path)
    kwargs = dict(config.get("model", {}).get("kwargs", {}))
    data_config = dict(config.get("data", {}))
    runtime_config = dict(config.get("runtime", {}))
    fit_X_train, fit_y_train, train_subsampled = _sample_xy_rows(
        X_train,
        y_train,
        data_config.get("max_train_rows"),
        random_state=random_state,
    )

    if kwargs.pop("evolutionary_algorithm", "AgeFitnessEA") == "AgeFitnessEA":
        kwargs["evolutionary_algorithm"] = AgeFitnessEA
    _set_default_random_state(kwargs, random_state)

    model = _instantiate_with_kwargs(SymbolicRegressor, kwargs)
    fit_start = time.time()
    with _time_limit(runtime_config.get("max_fit_seconds")):
        model.fit(fit_X_train, fit_y_train)
    fit_sec = time.time() - fit_start
    expr = _bingo_best_expr(model, feature_cols)
    best_pop = getattr(model, "best_pop", None)
    metrics = _metric_dict(
        "bingo",
        expr,
        model.predict(X_train),
        model.predict(X_val),
        model.predict(X_test),
        y_train,
        y_val,
        y_test,
        feature_cols,
        fit_sec,
        num_candidates=len(best_pop) if best_pop is not None else None,
    )
    metrics.update({
        "fit_n_train": int(len(fit_X_train)),
        "fit_train_subsampled": bool(train_subsampled),
        "max_fit_seconds": runtime_config.get("max_fit_seconds"),
    })
    return metrics


def fit_rils_rols(train_df, val_df, test_df, config_path, random_state=None):
    from rils_rols.rils_rols import RILSROLSRegressor

    X_train, y_train, feature_cols = dataframe_to_xy(train_df)
    X_val, y_val, _ = dataframe_to_xy(val_df)
    X_test, y_test, _ = dataframe_to_xy(test_df)
    config = load_yaml_config(config_path)
    kwargs = dict(config.get("model", {}).get("kwargs", {}))
    data_config = dict(config.get("data", {}))
    runtime_config = dict(config.get("runtime", {}))
    fit_X_train, fit_y_train, train_subsampled = _sample_xy_rows(
        X_train,
        y_train,
        data_config.get("max_train_rows"),
        random_state=random_state,
    )
    _set_default_random_state(kwargs, random_state)

    model = _instantiate_with_kwargs(RILSROLSRegressor, kwargs)
    fit_start = time.time()
    with _time_limit(runtime_config.get("max_fit_seconds")):
        model.fit(fit_X_train, fit_y_train)
    fit_sec = time.time() - fit_start
    expr = str(model.model_string())
    metrics = _metric_dict(
        "rils_rols",
        expr,
        model.predict(X_train),
        model.predict(X_val),
        model.predict(X_test),
        y_train,
        y_val,
        y_test,
        feature_cols,
        fit_sec,
        num_candidates=int(kwargs.get("max_fit_calls", 0)) if kwargs.get("max_fit_calls") else None,
    )
    metrics.update({
        "fit_n_train": int(len(fit_X_train)),
        "fit_train_subsampled": bool(train_subsampled),
        "max_fit_seconds": runtime_config.get("max_fit_seconds"),
    })
    return metrics


def fit_psrn(train_df, val_df, test_df, config_path, random_state=None):
    import torch
    from psrn import PSRN_Regressor
    import psrn.model.regressor as psrn_regressor_module

    X_train, y_train, feature_cols = dataframe_to_xy(train_df)
    X_val, y_val, _ = dataframe_to_xy(val_df)
    X_test, y_test, _ = dataframe_to_xy(test_df)
    config = load_yaml_config(config_path)
    model_config = dict(config.get("model", {}))
    fit_config = dict(config.get("fit", {}))
    runtime_config = dict(config.get("runtime", {}))

    max_train_rows = runtime_config.get("max_train_rows")
    fit_X_train, fit_y_train, train_subsampled = _sample_xy_rows(
        X_train,
        y_train,
        max_train_rows,
        random_state=random_state,
    )

    operators = list(model_config.get("operators", ["Add", "Mul", "Sub", "Div", "Identity"]))
    n_symbol_layers = int(model_config.get("n_symbol_layers", 2))
    n_inputs = int(model_config.get("n_inputs", max(3, min(6, fit_X_train.shape[1] + 2))))
    n_sample_variables = int(model_config.get("n_sample_variables", min(3, fit_X_train.shape[1])))
    use_const = bool(model_config.get("use_const", True))
    use_dr_mask = bool(model_config.get("use_dr_mask", False))
    token_generator = str(model_config.get("token_generator", "GP"))
    device_name = str(model_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    stage_config = {
        "default": {
            "operators": operators,
            "time_limit": int(fit_config.get("time_limit", 20)),
            "n_psrn_inputs": n_inputs,
            "n_sample_variables": n_sample_variables,
        },
        "stages": list(fit_config.get("stages", [{}])),
    }
    token_generator_config = dict(
        fit_config.get(
            "token_generator_config",
            {
                "base": {
                    "has_const": use_const,
                    "tokens": operators,
                }
            },
        )
    )

    variables = [f"x{i}" for i in range(fit_X_train.shape[1])]
    torch.manual_seed(int(random_state or 0))
    np.random.seed(int(random_state or 0))

    regressor = PSRN_Regressor(
        variables=variables,
        operators=operators,
        n_symbol_layers=n_symbol_layers,
        n_inputs=n_inputs,
        use_dr_mask=use_dr_mask,
        use_const=use_const,
        n_sample_variables=n_sample_variables,
        stage_config=stage_config,
        token_generator_config=token_generator_config,
        token_generator=token_generator,
        device=device,
    )
    tracker = SearchEfficiencyTracker("psrn")
    original_pareto_update = regressor.pareto_update_and_check

    def traced_pareto_update(new_samples):
        should_stop = original_pareto_update(new_samples)
        for candidate_rank, candidate in enumerate(new_samples, start=1):
            expr = str(candidate[0])
            try:
                pred = _predict_sympy_candidate(
                    expr,
                    X_val,
                    [f"x{i}" for i in range(X_val.shape[1])],
                )
                native_evaluations = max(
                    tracker.observed_evaluations + 1,
                    len(getattr(regressor, "fitted_expr_c_set", set())),
                )
                hit = tracker.observe(
                    expr,
                    (y_val, pred),
                    stage="fitted_pareto_candidate",
                    native_evaluations=native_evaluations,
                    extra={
                        "candidate_batch_rank": candidate_rank,
                        "training_mse": candidate[2] if len(candidate) > 2 else None,
                        "complexity": candidate[3] if len(candidate) > 3 else None,
                    },
                )
            except Exception:
                continue
            if hit:
                raise ValidationSearchHit
        return should_stop

    regressor.pareto_update_and_check = traced_pareto_update

    def _sanitize_psrn_expr_text(value):
        if not isinstance(value, str):
            return value

        def repl(match):
            exponent = str(int(match.group(1)))
            return f")**{exponent}"

        return re.sub(r"\)(\d+)", repl, value)

    sanitize_psrn_powers = bool(fit_config.get("sanitize_power_syntax", True))
    orig_sympy_s = getattr(psrn_regressor_module.sympy, "S", None)
    orig_sympify = getattr(psrn_regressor_module.sympy, "sympify", None)

    if sanitize_psrn_powers and orig_sympy_s is not None and orig_sympify is not None:
        def _safe_sympy_s(value, *args, **kwargs):
            return orig_sympy_s(_sanitize_psrn_expr_text(value), *args, **kwargs)

        def _safe_sympify(value, *args, **kwargs):
            return orig_sympify(_sanitize_psrn_expr_text(value), *args, **kwargs)

        psrn_regressor_module.sympy.S = _safe_sympy_s
        psrn_regressor_module.sympy.sympify = _safe_sympify

    fit_start = time.time()
    fit_error = None
    try:
        try:
            flag, pareto_frontier = regressor.fit(
                fit_X_train,
                fit_y_train.reshape(-1, 1),
                n_down_sample=int(fit_config.get("n_down_sample", min(50, len(fit_X_train)))),
                eta=float(fit_config.get("eta", 0.99)),
                use_threshold=bool(fit_config.get("use_threshold", False)),
                threshold=float(fit_config.get("threshold", 1e-10)),
                probe=None,
                prun_const=bool(fit_config.get("prun_const", True)),
                prun_ndigit=int(fit_config.get("prun_ndigit", 6)),
                real_time_display=bool(fit_config.get("real_time_display", False)),
                real_time_display_freq=int(fit_config.get("real_time_display_freq", 1)),
                real_time_display_ntop=int(fit_config.get("real_time_display_ntop", 10)),
                add_bias=bool(fit_config.get("add_bias", True)),
                together=bool(fit_config.get("together", False)),
                top_k=int(fit_config.get("top_k", 10)),
                use_replace_expo=bool(fit_config.get("use_replace_expo", False)),
                use_strict_pareto=bool(fit_config.get("use_strict_pareto", True)),
                use_extra_const=bool(fit_config.get("use_extra_const", False)),
            )
        except Exception as exc:
            fit_error = repr(exc)
            flag = False
            pareto_frontier = list(getattr(regressor, "pareto_frontier", []) or [])
        fit_sec = time.time() - fit_start
        pareto_by_mse = regressor.get_pf(sort_by="mse")
        if not pareto_by_mse and pareto_frontier:
            pareto_by_mse = list(pareto_frontier)
            pareto_by_mse.sort(key=lambda x: x[2] if len(x) > 2 else float("inf"))
        if not pareto_by_mse:
            raise RuntimeError(f"PSRN returned an empty Pareto frontier; fit_error={fit_error}")
        metrics = None
        skipped_expr_errors = []
        original_expression = getattr(regressor, "expression", None)
        selection_candidates = list(pareto_by_mse)
        if tracker.first_hit:
            hit_expr = tracker.first_hit.get("expression")
            selection_candidates = [
                (hit_expr, None, None, None),
                *[candidate for candidate in selection_candidates if str(candidate[0]) != hit_expr],
            ]
        for candidate in selection_candidates:
            expr = str(candidate[0])
            try:
                regressor.expression = expr
                candidate_metrics = _metric_dict(
                    "psrn",
                    expr,
                    _predict_sympy_candidate(
                        expr, X_train, [f"x{i}" for i in range(X_train.shape[1])]
                    ),
                    _predict_sympy_candidate(
                        expr, X_val, [f"x{i}" for i in range(X_val.shape[1])]
                    ),
                    _predict_sympy_candidate(
                        expr, X_test, [f"x{i}" for i in range(X_test.shape[1])]
                    ),
                    y_train,
                    y_val,
                    y_test,
                    feature_cols,
                    fit_sec,
                    num_candidates=len(pareto_frontier),
                )
                candidate_metrics["psrn_selected_pf_rank"] = int(len(skipped_expr_errors))
                metrics = candidate_metrics
                break
            except Exception as exc:
                skipped_expr_errors.append({"expr": expr, "error": repr(exc)})
        if metrics is None:
            if original_expression is not None:
                regressor.expression = original_expression
            raise RuntimeError(f"PSRN returned no evaluable Pareto expression: {skipped_expr_errors[:5]}")
    finally:
        if sanitize_psrn_powers and orig_sympy_s is not None and orig_sympify is not None:
            psrn_regressor_module.sympy.S = orig_sympy_s
            psrn_regressor_module.sympy.sympify = orig_sympify
    if skipped_expr_errors:
        metrics["psrn_skipped_invalid_expr_count"] = int(len(skipped_expr_errors))
        metrics["psrn_skipped_invalid_expr_examples"] = json.dumps(skipped_expr_errors[:5], ensure_ascii=False)
    if fit_error:
        metrics["psrn_fit_error_recovered"] = fit_error
    metrics.update({
        "fit_n_train": int(len(fit_X_train)),
        "fit_train_subsampled": bool(train_subsampled),
        "psrn_found_threshold": bool(flag),
        "psrn_device": str(device),
        "psrn_time_limit": int(stage_config["default"]["time_limit"]),
        "psrn_expr_power_sanitizer": bool(sanitize_psrn_powers),
        "validation_search_count_semantics": (
            "constant-fitted PSRN/PSE candidates entering Pareto evaluation; "
            "native tensor enumeration size is reported separately when available"
        ),
    })
    metrics.update(tracker.metrics())
    return metrics


def _deep_update_dict(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base


def fit_dso(train_df, val_df, test_df, config_path, random_state=None):
    # Import datasets before dso/tensorflow in Python 3.7 environments. TF1 can
    # leave tensorflow.__spec__ unset, which breaks datasets' availability check.
    try:
        import datasets  # noqa: F401
    except ImportError:
        pass
    from dso import DeepSymbolicRegressor

    X_train, y_train, feature_cols = dataframe_to_xy(train_df)
    X_val, y_val, _ = dataframe_to_xy(val_df)
    X_test, y_test, _ = dataframe_to_xy(test_df)
    config = load_yaml_config(config_path)
    runtime_config = dict(config.get("runtime", {}))
    max_train_rows = runtime_config.get("max_train_rows")
    fit_X_train, fit_y_train, train_subsampled = _sample_xy_rows(
        X_train,
        y_train,
        max_train_rows,
        random_state=random_state,
    )

    dso_config = {
        "experiment": {
            "logdir": None,
            "seed": int(random_state or 0),
        },
        "task": {
            "task_type": "regression",
            "function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"],
            "metric": "inv_nrmse",
            "metric_params": [1.0],
            "protected": True,
            "threshold": 1e-6,
        },
        "training": {
            "n_samples": 1000,
            "batch_size": 200,
            "epsilon": 0.1,
            "n_cores_batch": 1,
            "verbose": False,
            "early_stopping": True,
        },
        "policy": {
            "max_length": 24,
            "num_layers": 1,
            "num_units": 32,
        },
        "policy_optimizer": {
            "learning_rate": 0.001,
            "entropy_weight": 0.005,
            "entropy_gamma": 1.0,
            "summary": False,
        },
        "prior": {
            "length": {
                "min_": 2,
                "max_": 24,
                "on": True,
            },
            "inverse": {"on": True},
            "trig": {"on": True},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": True},
            "soft_length": {
                "loc": 8,
                "scale": 4,
                "on": True,
            },
        },
        "logging": {
            "save_all_iterations": False,
            "save_summary": False,
            "save_positional_entropy": False,
            "save_pareto_front": False,
            "save_cache": False,
            "hof": 20,
        },
        "gp_meld": {
            "run_gp_meld": False,
        },
    }
    _deep_update_dict(dso_config, dict(config.get("dso", {})))

    dso_config["task"]["dataset"] = (fit_X_train, fit_y_train)
    model = DeepSymbolicRegressor(dso_config)
    tracker = SearchEfficiencyTracker("dso")
    fit_start = time.time()
    max_fit_seconds = runtime_config.get("max_fit_seconds")
    hit_program = None
    seen_cache_keys = set()
    with _time_limit(max_fit_seconds):
        model.setup()
        from dso.program import Program

        while not model.trainer.done:
            model.train_one_step()
            native_evaluations = int(model.trainer.nevals)
            for cache_key, program in list(Program.cache.items()):
                if cache_key in seen_cache_keys:
                    continue
                seen_cache_keys.add(cache_key)
                try:
                    pred = np.asarray(program.execute(X_val), dtype=float).reshape(-1)
                    hit = tracker.observe(
                        str(getattr(program, "sympy_expr", None) or program.pretty()),
                        (y_val, pred),
                        stage=f"batch_{model.trainer.iteration}",
                        native_evaluations=native_evaluations,
                        extra={
                            "training_iteration": int(model.trainer.iteration),
                            "program_reward": getattr(program, "r", None),
                        },
                    )
                except Exception:
                    continue
                if hit:
                    hit_program = program
                    break
            if hit_program is not None:
                break
    if hit_program is None:
        hit_program = model.trainer.p_r_best
    model.program_ = hit_program
    if not model.trainer.done and model.pool is not None:
        model.pool.close()
    fit_sec = time.time() - fit_start
    expr = str(getattr(model.program_, "sympy_expr", None) or model.program_.pretty())
    metrics = _metric_dict(
        "dso",
        expr,
        model.predict(X_train),
        model.predict(X_val),
        model.predict(X_test),
        y_train,
        y_val,
        y_test,
        feature_cols,
        fit_sec,
        num_candidates=int(dso_config.get("training", {}).get("n_samples", 0)),
    )
    metrics.update({
        "fit_n_train": int(len(fit_X_train)),
        "fit_train_subsampled": bool(train_subsampled),
        "max_fit_seconds": max_fit_seconds,
        "validation_search_count_semantics": (
            "all newly cached DSO programs checked after each sampled batch; "
            "native hit count is the trainer nevals value at the batch boundary"
        ),
    })
    metrics.update(tracker.metrics())
    return metrics


def _pyoperon_best_expr(model, feature_cols):
    model_tree = getattr(model, "model_", None)
    get_model_string = getattr(model, "get_model_string", None)
    if callable(get_model_string) and model_tree is not None:
        try:
            return str(get_model_string(model_tree, names=[str(x) for x in feature_cols]))
        except Exception:
            try:
                return str(get_model_string(model_tree))
            except Exception:
                pass
    for attr in ["get_model", "model_string_", "model_", "program_", "_program"]:
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        try:
            value = obj() if callable(obj) else obj
            if value is not None:
                return str(value)
        except Exception:
            continue
    return str(model)


def fit_pyoperon(train_df, val_df, test_df, config_path, random_state=None):
    from pyoperon.sklearn import SymbolicRegressor as OperonSymbolicRegressor

    X_train, y_train, feature_cols = dataframe_to_xy(train_df)
    X_val, y_val, _ = dataframe_to_xy(val_df)
    X_test, y_test, _ = dataframe_to_xy(test_df)
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    y_train = np.ascontiguousarray(y_train, dtype=np.float64)
    X_val = np.ascontiguousarray(X_val, dtype=np.float64)
    y_val = np.ascontiguousarray(y_val, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    y_test = np.ascontiguousarray(y_test, dtype=np.float64)
    config = load_yaml_config(config_path)
    kwargs = dict(config.get("model", {}).get("kwargs", config.get("fit", {})))
    data_config = dict(config.get("data", {}))
    runtime_config = dict(config.get("runtime", {}))
    _set_default_random_state(kwargs, random_state)
    fit_X_train, fit_y_train, train_subsampled = _sample_xy_rows(
        X_train,
        y_train,
        data_config.get("max_train_rows"),
        random_state=random_state,
    )
    fit_X_train = np.ascontiguousarray(fit_X_train, dtype=np.float64)
    fit_y_train = np.ascontiguousarray(fit_y_train, dtype=np.float64)

    model = _instantiate_with_kwargs(OperonSymbolicRegressor, kwargs)
    fit_start = time.time()
    with _time_limit(runtime_config.get("max_fit_seconds")):
        model.fit(fit_X_train, fit_y_train)
    fit_sec = time.time() - fit_start
    expr = _pyoperon_best_expr(model, feature_cols)
    pareto_front = getattr(model, "pareto_front_", None)
    num_candidates = len(pareto_front) if pareto_front is not None else None
    metrics = _metric_dict(
        "pyoperon",
        expr,
        model.predict(X_train),
        model.predict(X_val),
        model.predict(X_test),
        y_train,
        y_val,
        y_test,
        feature_cols,
        fit_sec,
        num_candidates=num_candidates,
    )
    metrics.update({
        "fit_n_train": int(len(fit_X_train)),
        "fit_train_subsampled": bool(train_subsampled),
        "max_fit_seconds": runtime_config.get("max_fit_seconds"),
    })
    return metrics


def _itea_tfuncs(names):
    try:
        import jax.numpy as jnp
    except Exception:
        jnp = np

    funcs = {
        "id": lambda x: x,
        "sin": jnp.sin,
        "cos": jnp.cos,
        "exp": jnp.exp,
        "log": lambda x: jnp.log(jnp.abs(x) + 1.0e-12),
        "sqrt": lambda x: jnp.sqrt(jnp.abs(x)),
    }
    return {name: funcs[name] for name in names if name in funcs}


def _patch_itea_for_sklearn_metrics():
    import inspect
    from sklearn.base import BaseEstimator
    import sklearn.metrics
    import itea.inspection._ITExpr_explainer as itexpr_explainer
    import itea.regression._ITExpr_regressor as itexpr_regressor

    explainer_cls = getattr(itexpr_explainer, "ITExpr_explainer", None)
    if explainer_cls is not None and not hasattr(explainer_cls, "__sklearn_tags__"):
        explainer_cls.__sklearn_tags__ = lambda self: BaseEstimator().__sklearn_tags__()

    if "squared" in inspect.signature(sklearn.metrics.mean_squared_error).parameters:
        return

    def mean_squared_error_compat(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", squared=True):
        value = sklearn.metrics.mean_squared_error(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )
        return value if squared else float(np.sqrt(value))

    itexpr_regressor.mean_squared_error = mean_squared_error_compat


def _itea_best_expr(model):
    best = getattr(model, "bestsol_", None)
    if best is None:
        return str(model)
    for attr in ["to_str", "to_string"]:
        obj = getattr(best, attr, None)
        if callable(obj):
            try:
                return str(obj())
            except Exception:
                pass
    return str(best)


def fit_itea(train_df, val_df, test_df, config_path, random_state=None):
    from itea.regression import ITEA_regressor

    _patch_itea_for_sklearn_metrics()
    X_train, y_train, feature_cols = dataframe_to_xy(train_df)
    X_val, y_val, _ = dataframe_to_xy(val_df)
    X_test, y_test, _ = dataframe_to_xy(test_df)
    config = load_yaml_config(config_path)
    kwargs = dict(config.get("model", {}).get("kwargs", config.get("fit", {})))
    data_config = dict(config.get("data", {}))
    runtime_config = dict(config.get("runtime", {}))
    tfunc_names = kwargs.pop("tfuncs", ["id"])
    if isinstance(tfunc_names, dict):
        tfuncs = tfunc_names
    else:
        tfuncs = _itea_tfuncs(tfunc_names)
    kwargs["tfuncs"] = tfuncs
    kwargs["labels"] = [str(x) for x in feature_cols]
    _set_default_random_state(kwargs, random_state)
    if isinstance(kwargs.get("expolim"), list):
        kwargs["expolim"] = tuple(kwargs["expolim"])

    fit_X_train, fit_y_train, train_subsampled = _sample_xy_rows(
        X_train,
        y_train,
        data_config.get("max_train_rows"),
        random_state=random_state,
    )
    max_fit_seconds = runtime_config.get("max_fit_seconds")

    model = _instantiate_with_kwargs(ITEA_regressor, kwargs)
    fit_start = time.time()
    with _time_limit(max_fit_seconds):
        model.fit(fit_X_train, fit_y_train)
    fit_sec = time.time() - fit_start
    expr = _itea_best_expr(model)
    metrics = _metric_dict(
        "itea",
        expr,
        model.predict(X_train),
        model.predict(X_val),
        model.predict(X_test),
        y_train,
        y_val,
        y_test,
        feature_cols,
        fit_sec,
        num_candidates=int(kwargs.get("popsize", 0)) * int(kwargs.get("gens", 0)) if kwargs.get("popsize") and kwargs.get("gens") else None,
    )
    metrics.update({
        "fit_n_train": int(len(fit_X_train)),
        "fit_train_subsampled": bool(train_subsampled),
        "max_fit_seconds": max_fit_seconds,
    })
    return metrics


def _parse_aifeynman_solution(solution_path):
    if not os.path.exists(solution_path) or os.path.getsize(solution_path) == 0:
        return None, 0
    candidates = []
    with open(solution_path, "r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) >= 6:
                expr = "".join(parts[5:])
                if expr:
                    candidates.append(expr)
    return (candidates[-1] if candidates else None), len(candidates)


def fit_aifeynman(train_df, val_df, test_df, config_path, random_state=None):
    X_train, y_train, feature_cols = dataframe_to_xy(train_df)
    X_val, y_val, _ = dataframe_to_xy(val_df)
    X_test, y_test, _ = dataframe_to_xy(test_df)
    config = load_yaml_config(config_path)
    run_config = dict(config.get("run", {}))
    data_config = dict(config.get("data", {}))
    fit_X_train, fit_y_train, train_subsampled = _sample_xy_rows(
        X_train,
        y_train,
        data_config.get("max_train_rows"),
        random_state=random_state,
    )

    runner = PROJECT_ROOT / "srsd-benchmark" / "external" / "ai_feynman2" / "ai_feynman_runner.py"
    max_fit_seconds = float(run_config.get("max_fit_seconds", 900))
    with tempfile.TemporaryDirectory(prefix="aifeynman_", dir=os.environ.get("CPU_SR_TMPDIR", "/tmp")) as tmpdir:
        case_path = os.path.join(tmpdir, "case.txt")
        np.savetxt(case_path, np.column_stack([fit_X_train, fit_y_train]))
        cmd = [
            sys.executable,
            str(runner),
            "--src",
            case_path,
            "--op",
            str(run_config.get("op", "7ops.txt")),
            "--epoch",
            str(int(run_config.get("epoch", 1))),
            "--bftt",
            str(int(run_config.get("bftt", 1))),
            "--poly_deg",
            str(int(run_config.get("poly_deg", 2))),
        ]
        fit_start = time.time()
        completed = subprocess.run(
            cmd,
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max_fit_seconds,
            check=False,
        )
        fit_sec = time.time() - fit_start
        solution_path = os.path.join(tmpdir, "results", "solution_case.txt")
        expr, num_candidates = _parse_aifeynman_solution(solution_path)
        if not expr:
            raise RuntimeError(
                f"AI Feynman found no expression; returncode={completed.returncode}; "
                f"tail={completed.stdout[-1000:] if completed.stdout else ''}"
            )
        metrics = _metric_dict(
            "aifeynman",
            expr,
            evaluate_expression(expr, train_df, feature_cols),
            evaluate_expression(expr, val_df, feature_cols),
            evaluate_expression(expr, test_df, feature_cols),
            y_train,
            y_val,
            y_test,
            feature_cols,
            fit_sec,
            num_candidates=num_candidates,
        )
        metrics.update({
            "fit_n_train": int(len(fit_X_train)),
            "fit_train_subsampled": bool(train_subsampled),
            "max_fit_seconds": max_fit_seconds,
        })
        return metrics


def _protected_div(left, right):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.where(np.abs(right) > 1.0e-12, left / right, 1.0)


def _protected_log(x):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.log(np.abs(x) + 1.0e-12)


def _protected_sqrt(x):
    return np.sqrt(np.abs(x))


def _protected_exp(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.clip(np.exp(np.clip(x, -30, 30)), -1.0e12, 1.0e12)


def fit_deap(train_df, val_df, test_df, config_path, random_state=None):
    import operator
    import random
    from deap import algorithms, base, creator, gp, tools

    X_train, y_train, feature_cols = dataframe_to_xy(train_df)
    X_val, y_val, _ = dataframe_to_xy(val_df)
    X_test, y_test, _ = dataframe_to_xy(test_df)
    config = load_yaml_config(config_path)
    kwargs = dict(config.get("model", {}).get("kwargs", {}))
    data_config = dict(config.get("data", {}))
    runtime_config = dict(config.get("runtime", {}))
    fit_X_train, fit_y_train, train_subsampled = _sample_xy_rows(
        X_train,
        y_train,
        data_config.get("max_train_rows"),
        random_state=random_state,
    )

    seed = int(random_state if random_state is not None else kwargs.get("random_state", 42))
    random.seed(seed)
    np.random.seed(seed)

    pset = gp.PrimitiveSet("MAIN", len(feature_cols))
    pset.renameArguments(**{f"ARG{i}": str(name) for i, name in enumerate(feature_cols)})
    ops = set(kwargs.get("operators", ["add", "sub", "mul", "div", "sin", "cos", "log", "sqrt"]))
    if "add" in ops:
        pset.addPrimitive(operator.add, 2, name="add")
    if "sub" in ops:
        pset.addPrimitive(operator.sub, 2, name="sub")
    if "mul" in ops:
        pset.addPrimitive(operator.mul, 2, name="mul")
    if "div" in ops:
        pset.addPrimitive(_protected_div, 2, name="div")
    if "sin" in ops:
        pset.addPrimitive(np.sin, 1, name="sin")
    if "cos" in ops:
        pset.addPrimitive(np.cos, 1, name="cos")
    if "log" in ops:
        pset.addPrimitive(_protected_log, 1, name="log")
    if "sqrt" in ops:
        pset.addPrimitive(_protected_sqrt, 1, name="sqrt")
    if "exp" in ops:
        pset.addPrimitive(_protected_exp, 1, name="exp")
    pset.addEphemeralConstant("rand", lambda: random.uniform(-5.0, 5.0))

    if not hasattr(creator, "FitnessMinSR"):
        creator.create("FitnessMinSR", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualSR"):
        creator.create("IndividualSR", gp.PrimitiveTree, fitness=creator.FitnessMinSR)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=int(kwargs.get("init_max_depth", 3)))
    toolbox.register("individual", tools.initIterate, creator.IndividualSR, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evaluate_individual(individual):
        try:
            func = toolbox.compile(expr=individual)
            pred = func(*[fit_X_train[:, i] for i in range(fit_X_train.shape[1])])
            pred = _coerce_predictions(pred)
            if len(pred) != len(fit_y_train):
                return (1.0e30,)
            mse = safe_mse(fit_y_train, pred)
            if mse is None or not np.isfinite(mse):
                return (1.0e30,)
            return (float(mse) + float(kwargs.get("parsimony", 1.0e-4)) * len(individual),)
        except Exception:
            return (1.0e30,)

    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("select", tools.selTournament, tournsize=int(kwargs.get("tournament_size", 3)))
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=int(kwargs.get("mut_max_depth", 2)))
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    max_height = int(kwargs.get("max_height", 8))
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))

    pop = toolbox.population(n=int(kwargs.get("population_size", 300)))
    hall = tools.HallOfFame(1)
    fit_start = time.time()
    with _time_limit(runtime_config.get("max_fit_seconds")):
        algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=float(kwargs.get("cxpb", 0.5)),
            mutpb=float(kwargs.get("mutpb", 0.2)),
            ngen=int(kwargs.get("generations", 30)),
            halloffame=hall,
            verbose=False,
        )
    fit_sec = time.time() - fit_start
    best = hall[0] if len(hall) else tools.selBest(pop, 1)[0]
    expr = str(best)
    func = toolbox.compile(expr=best)
    metrics = _metric_dict(
        "deap",
        expr,
        func(*[X_train[:, i] for i in range(X_train.shape[1])]),
        func(*[X_val[:, i] for i in range(X_val.shape[1])]),
        func(*[X_test[:, i] for i in range(X_test.shape[1])]),
        y_train,
        y_val,
        y_test,
        feature_cols,
        fit_sec,
        num_candidates=int(kwargs.get("population_size", 300)) * int(kwargs.get("generations", 30)),
    )
    metrics.update({
        "fit_n_train": int(len(fit_X_train)),
        "fit_train_subsampled": bool(train_subsampled),
        "max_fit_seconds": runtime_config.get("max_fit_seconds"),
    })
    return metrics


METHOD_FITTERS = {
    "aifeynman": fit_aifeynman,
    "bingo": fit_bingo,
    "deap": fit_deap,
    "dso": fit_dso,
    "ffx": fit_ffx,
    "gplearn": fit_gplearn,
    "psrn": fit_psrn,
    "pysr": fit_pysr,
    "pyoperon": fit_pyoperon,
    "rils_rols": fit_rils_rols,
    "itea": fit_itea,
}


def collect_tasks(benchmark: str):
    if benchmark == "sldbench":
        mod = import_script_module("cpu_sldbench_loader", SCRIPT_DIR / "run_sldbench.py")
        return mod, mod.collect_sldbench_tasks()
    if benchmark == "llmsrbench":
        mod = import_script_module("cpu_llmsrbench_loader", SCRIPT_DIR / "run_llmsrbench.py")
        if not mod.LLMSRBENCH_CASES_ROOT and not os.path.exists(mod.LLMSRBENCH_HDF5):
            raise FileNotFoundError(f"hdf5 not found: {mod.LLMSRBENCH_HDF5}")
        return mod, mod.collect_llmsrbench_tasks()
    if benchmark == "srbench":
        mod = import_script_module("cpu_srbench_loader", SCRIPT_DIR / "run_SRbench.py")
        return mod, mod.collect_srbench_tasks()
    if benchmark == "srsd":
        mod = import_script_module("cpu_srsd_loader", SCRIPT_DIR / "run_srds.py")
        frames = []
        for dataset_dir_name in mod.SRSD_DATASET_DIRS:
            print("\n" + "#" * 100)
            print(f"SCAN DATASET DIR: {dataset_dir_name}")
            print("#" * 100)
            df = mod.collect_raw_tasks_for_dataset(dataset_dir_name)
            if len(df):
                frames.append(df)
        if frames:
            return mod, pd.concat(frames, axis=0, ignore_index=True)
        return mod, pd.DataFrame()
    raise ValueError(f"unknown benchmark: {benchmark}")


def load_case(benchmark: str, mod, row):
    if benchmark == "sldbench":
        train_df, val_df, test_df = mod.load_sldbench_case(row)
        meta = {
            "task_type": "sldbench",
            "dataset_dir": "sldbench",
            "difficulty": row.get("task_name"),
            "base_name": row.get("case_name"),
            "case_name": row.get("case_name"),
            "sldbench_task_name": row.get("task_name"),
            "sldbench_group_key": row.get("group_key"),
            "sldbench_target_idx": row.get("target_idx"),
            "sldbench_target_name": row.get("target_name"),
            "original_feature_names": " | ".join([str(x) for x in row.get("feature_names", [])]),
        }
        return train_df, val_df, test_df, meta

    if benchmark == "llmsrbench":
        train_df, val_df, test_df, extra = mod.load_llmsrbench_case(row)
        meta = {
            "task_type": "llmsrbench",
            "dataset_dir": "llmsrbench",
            "difficulty": row.get("split_name"),
            "base_name": row.get("case_name"),
            "case_name": row.get("case_name"),
            "true_expression": row.get("true_expression"),
            "llmsrbench_split_name": row.get("split_name"),
            "llmsrbench_case_name": row.get("case_name"),
            "original_symbols": " | ".join([str(x) for x in row.get("symbols", [])]),
            "feature_names_original": " | ".join([str(x) for x in (extra.get("feature_names_original") or [])]),
            "target_name_original": extra.get("target_name_original"),
            "layout_source": extra.get("layout_source"),
        }
        return train_df, val_df, test_df, meta

    if benchmark == "srbench":
        raw_df, source_kind, source_path = mod.load_raw_dataset_frame(str(row["dataset_name"]))
        xy_df, extra_info = mod.make_numeric_xy_dataframe(raw_df)
        train_df, val_df, test_df = mod.split_train_val_test(
            xy_df,
            test_ratio=mod.TEST_RATIO,
            val_ratio_within_trainval=mod.VAL_RATIO_WITHIN_TRAINVAL,
            seed=mod.TRAIN_TEST_SPLIT_SEED,
            shuffle=mod.SPLIT_SHUFFLE,
        )
        meta = {
            "task_type": "srbench",
            "dataset_dir": "srbench",
            "difficulty": row.get("group_name"),
            "base_name": row.get("dataset_name"),
            "case_name": row.get("dataset_name"),
            "srbench_group_name": row.get("group_name"),
            "srbench_dataset_name": row.get("dataset_name"),
            "data_source_kind": source_kind,
            "data_source_path": source_path,
            "original_feature_names": " | ".join(extra_info.get("original_feature_names", [])),
            "sanitized_feature_names": " | ".join(extra_info.get("sanitized_feature_names", [])),
        }
        return train_df, val_df, test_df, meta

    if benchmark == "srsd":
        train_df = mod.load_txt_dataset(row["train_path"])
        val_df = mod.load_txt_dataset(row["val_path"])
        test_df = mod.load_txt_dataset(row["test_path"])
        meta = {
            "task_type": "srsd",
            "dataset_dir": row.get("dataset_dir"),
            "difficulty": row.get("difficulty"),
            "base_name": row.get("base_name"),
            "case_name": row.get("base_name"),
            "true_expression": row.get("true_expression"),
            "train_path": row.get("train_path"),
            "val_path": row.get("val_path"),
            "test_path": row.get("test_path"),
            "true_eq_path": row.get("true_eq_path"),
        }
        return train_df, val_df, test_df, meta

    raise ValueError(f"unknown benchmark: {benchmark}")


def row_case_name(benchmark: str, row):
    if benchmark == "sldbench":
        return row.get("case_name")
    if benchmark == "llmsrbench":
        return row.get("case_name")
    if benchmark == "srbench":
        return row.get("dataset_name")
    if benchmark == "srsd":
        return row.get("base_name")
    return row.get("case_name", "case")


def json_safe_value(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if not isinstance(value, (list, tuple, dict, set)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe_value(v) for v in value]
    return value


def json_safe_record(record):
    return {str(k): json_safe_value(v) for k, v in dict(record).items()}


def failure_result(benchmark: str, method: str, row, runtime_sec: float, error: str):
    return {
        "method": method,
        "task_type": benchmark,
        "dataset_dir": benchmark,
        "difficulty": row.get("task_name") or row.get("split_name") or row.get("group_name") or row.get("dataset_dir"),
        "base_name": row_case_name(benchmark, row),
        "case_name": row_case_name(benchmark, row),
        "n_features": row.get("n_features") or row.get("n_features_meta"),
        "n_train": row.get("n_train"),
        "n_val": row.get("n_val"),
        "n_test": row.get("n_test"),
        "valid_formula_found": False,
        "num_candidate_exprs": 0,
        "best_expr": None,
        "best_expr_sympy": None,
        "best_train_mse": None,
        "best_val_mse": None,
        "best_test_mse": None,
        "train_r2": None,
        "val_r2": None,
        "test_r2": None,
        "expr_complexity": None,
        "expr_depth": None,
        "expr_string_length": None,
        "expr_sympy_ops": None,
        "passed": False,
        "perfect_fit": False,
        "perfect_fit_by_r2": False,
        "runtime_sec": float(runtime_sec),
        "error": str(error),
    }


def run_one(benchmark: str, method: str, mod, row, config_path: str, random_state=None):
    start = time.time()
    try:
        train_df, val_df, test_df, meta = load_case(benchmark, mod, row)
        if method not in METHOD_FITTERS:
            raise ValueError(f"unsupported method for this runner: {method}")
        result = METHOD_FITTERS[method](
            train_df,
            val_df,
            test_df,
            config_path=config_path,
            random_state=random_state,
        )
        result.update(meta)
        result.update({
            "n_features": int(train_df.shape[1] - 1),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "runtime_sec": float(time.time() - start),
            "error": None,
        })
        return result
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        return failure_result(benchmark, method, row, time.time() - start, repr(e))


def estimate_case_subprocess_timeout(config_path: str):
    override = os.environ.get("CPU_SR_CASE_SUBPROCESS_TIMEOUT_SEC")
    if override is not None and str(override).strip():
        value = float(override)
        return None if value <= 0 else value

    budgets = []
    try:
        config = load_yaml_config(config_path)
    except Exception:
        config = {}

    for section, key in [
        ("runtime", "max_fit_seconds"),
        ("run", "max_fit_seconds"),
        ("fit", "timeout_in_seconds"),
        ("fit", "time_limit"),
    ]:
        value = dict(config.get(section, {})).get(key)
        if value:
            budgets.append(float(value))
    value = dict(config.get("model", {})).get("kwargs", {}).get("max_time")
    if value:
        budgets.append(float(value))

    if not budgets:
        return None
    return max(budgets) + float(os.environ.get("CPU_SR_CASE_SUBPROCESS_GRACE_SEC", "900"))


def read_log_tail(path: Path, max_chars: int = CASE_LOG_TAIL_CHARS):
    try:
        data = path.read_bytes()
    except Exception:
        return ""
    if max_chars > 0 and len(data) > max_chars:
        data = data[-max_chars:]
    return data.decode("utf-8", errors="replace")


def read_efficiency_trace_metrics(path: Path):
    if not path.exists():
        return {}
    opener = gzip.open if path.suffix == ".gz" else open
    last = None
    first_hit = None
    best = None
    try:
        with opener(path, "rt", encoding="utf-8") as fp:
            for line in fp:
                event = json.loads(line)
                last = event
                if event.get("validation_success") and first_hit is None:
                    first_hit = event
                r2 = event.get("validation_r2")
                if r2 is not None and (best is None or r2 > best.get("validation_r2", -math.inf)):
                    best = event
    except Exception:
        # A hard timeout may leave the final gzip footer unwritten. Events
        # yielded before the truncated tail are still valid observations.
        pass
    if last is None:
        return {}
    hit = first_hit or {}
    return {
        "validation_search_threshold": float(SEARCH_EFFICIENCY_THRESHOLD),
        "validation_search_rule": "strictly_greater",
        "validation_search_success": bool(first_hit),
        "evaluations_to_validation_success": hit.get("candidate_index"),
        "unique_evaluations_to_validation_success": hit.get("unique_candidate_index"),
        "native_evaluations_to_validation_success": hit.get("native_evaluations"),
        "first_validation_success_stage": hit.get("stage"),
        "first_validation_success_expression": hit.get("expression"),
        "first_validation_success_r2": hit.get("validation_r2"),
        "first_validation_success_elapsed_sec": hit.get("elapsed_sec"),
        "validation_search_observed_evaluations": last.get("candidate_index"),
        "validation_search_unique_evaluations": last.get("unique_candidate_index"),
        "validation_search_native_evaluations": last.get("native_evaluations"),
        "validation_search_best_r2": (best or {}).get("validation_r2"),
        "validation_search_best_expression": (best or {}).get("expression"),
        "validation_search_trace": str(path),
    }


def terminate_process_group(pgid: int, grace_sec: float = 10.0):
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        return
    deadline = time.time() + max(0.0, float(grace_sec))
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
    except Exception:
        pass


def run_logged_child(cmd, env, log_fp, timeout_sec):
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        returncode = proc.wait(timeout=timeout_sec)
        if returncode != 0:
            terminate_process_group(proc.pid, grace_sec=2.0)
        return returncode
    except subprocess.TimeoutExpired:
        terminate_process_group(proc.pid)
        raise


def run_single_case_child(args, mod, df_tasks, config_path: str):
    idx = int(args.single_case_index)
    if idx < 1 or idx > len(df_tasks):
        raise IndexError(f"single-case index {idx} out of range 1..{len(df_tasks)}")

    row = df_tasks.iloc[idx - 1].to_dict()
    print(f"[{idx}/{len(df_tasks)}] Starting: {row_case_name(args.benchmark, row)}", flush=True)
    res = run_one(
        benchmark=args.benchmark,
        method=args.method,
        mod=mod,
        row=row,
        config_path=config_path,
        random_state=args.random_state,
    )
    res["completed_index"] = int(idx)
    res["total_tasks"] = int(len(df_tasks))
    configured_budget = os.environ.get("CPU_SR_CONFIGURED_WALL_BUDGET_SEC")
    if configured_budget is not None:
        res["configured_wall_budget_sec"] = float(configured_budget)
    print_one_result(args.benchmark, idx, len(df_tasks), row, res)
    if args.single_result_json:
        with open(args.single_result_json, "w", encoding="utf-8") as fp:
            json.dump(json_safe_record(res), fp, ensure_ascii=False, allow_nan=False, indent=2)
    return res


def run_isolated_case(args, idx: int, total: int, row, config_path: str, results_root: str):
    start = time.time()
    case_name = row_case_name(args.benchmark, row)
    case_slug = sanitize_name(case_name or f"case_{idx}")
    case_log_dir = Path(results_root) / "case_logs"
    case_result_dir = Path(results_root) / "case_results"
    case_trace_dir = Path(results_root) / "candidate_traces"
    case_log_dir.mkdir(parents=True, exist_ok=True)
    case_result_dir.mkdir(parents=True, exist_ok=True)
    case_trace_dir.mkdir(parents=True, exist_ok=True)
    case_log = case_log_dir / f"{idx:04d}_{case_slug}.log"
    result_json = case_result_dir / f"{idx:04d}_{case_slug}.json"
    trace_path = case_trace_dir / f"{idx:04d}_{case_slug}.jsonl.gz"
    if args.resume and result_json.exists():
        with open(result_json, "r", encoding="utf-8") as fp:
            res = json.load(fp)
        res["resumed_existing_result"] = True
        return json_safe_record(res)
    if result_json.exists():
        result_json.unlink()

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--benchmark",
        args.benchmark,
        "--method",
        args.method,
        "--config",
        str(config_path),
        "--results-root",
        str(results_root),
        "--random-state",
        str(args.random_state),
        "--single-case-index",
        str(idx),
        "--single-result-json",
        str(result_json),
    ]
    if args.max_cases is not None and args.max_cases > 0:
        cmd.extend(["--max-cases", str(args.max_cases)])

    timeout_sec = estimate_case_subprocess_timeout(config_path)
    env = os.environ.copy()
    env["CPU_SR_CHILD_CASE"] = "1"
    env["CPU_SR_SEARCH_EFFICIENCY_TRACE"] = str(trace_path)

    print(f"   isolated_case_log:   {case_log}", flush=True)
    try:
        with open(case_log, "w", encoding="utf-8", errors="replace") as log_fp:
            returncode = run_logged_child(
                cmd,
                env,
                log_fp,
                timeout_sec,
            )
        if result_json.exists():
            with open(result_json, "r", encoding="utf-8") as fp:
                res = json.load(fp)
            if returncode != 0 and not res.get("error"):
                res["error"] = f"isolated child returned non-zero status {returncode}"
        else:
            tail = read_log_tail(case_log)
            res = failure_result(
                args.benchmark,
                args.method,
                row,
                time.time() - start,
                f"isolated child failed without result json; returncode={returncode}; log_tail={tail}",
            )
    except subprocess.TimeoutExpired:
        tail = read_log_tail(case_log)
        res = failure_result(
            args.benchmark,
            args.method,
            row,
            time.time() - start,
            f"isolated child exceeded outer timeout {timeout_sec:.1f}s; log_tail={tail}",
        )

    trace_metrics = read_efficiency_trace_metrics(trace_path)
    for key, value in trace_metrics.items():
        if res.get(key) is None:
            res[key] = value
    res["completed_index"] = int(idx)
    res["total_tasks"] = int(total)
    res["case_log_path"] = str(case_log)
    configured_budget = os.environ.get("CPU_SR_CONFIGURED_WALL_BUDGET_SEC")
    if configured_budget is not None:
        res["configured_wall_budget_sec"] = float(configured_budget)
    res = json_safe_record(res)
    # Child crashes and outer timeouts are valid censored observations.  They
    # must be persisted just like successful child results so --resume does
    # not rerun them and downstream success-rate denominators remain correct.
    tmp_result_json = result_json.with_suffix(result_json.suffix + ".tmp")
    with open(tmp_result_json, "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(tmp_result_json, result_json)
    return res


def print_one_result(benchmark: str, idx: int, total: int, row, res):
    print(f"[{idx}/{total}] Processing: {row_case_name(benchmark, row)}")
    if benchmark == "sldbench":
        print(f"   task_name:           {row.get('task_name')}")
        print(f"   group_key:           {row.get('group_key')}")
        print(f"   target_name:         {row.get('target_name')}")
        print(f"   original_features:   {row.get('feature_names')}")
    elif benchmark == "llmsrbench":
        print(f"   split_name:          {row.get('split_name')}")
        print(f"   true_expression:     {row.get('true_expression')}")
        print(f"   original_symbols:    {row.get('symbols')}")
    elif benchmark == "srbench":
        print(f"   group_name:          {row.get('group_name')}")
        print(f"   data_source_kind:    {res.get('data_source_kind')}")
    elif benchmark == "srsd":
        print(f"   dataset_dir:         {row.get('dataset_dir')}")
        print(f"   true_expression:     {row.get('true_expression')}")

    print(f"   method:              {res.get('method')}")
    print(f"   train/val/test:      ({res.get('n_train')}, {res.get('n_val')}, {res.get('n_test')})")
    print(f"   n_features:          {res.get('n_features')}")
    print(f"   valid_formula_found: {res.get('valid_formula_found')}")
    print(f"   num_candidate_exprs: {res.get('num_candidate_exprs')}")
    print(f"   best_expr:           {res.get('best_expr')}")
    print(f"   best_expr_sympy:     {res.get('best_expr_sympy')}")
    print(f"   expr_complexity:     {res.get('expr_complexity')}")
    print(f"   expr_depth:          {res.get('expr_depth')}")
    print(f"   expr_string_length:  {res.get('expr_string_length')}")
    print(f"   expr_sympy_ops:      {res.get('expr_sympy_ops')}")
    print(f"   best_train_mse:      {res.get('best_train_mse')}")
    print(f"   best_val_mse:        {res.get('best_val_mse')}")
    print(f"   best_test_mse:       {res.get('best_test_mse')}")
    print(f"   train_r2:            {res.get('train_r2')}")
    print(f"   val_r2:              {res.get('val_r2')}")
    print(f"   test_r2:             {res.get('test_r2')}")
    print(f"   passed:              {res.get('passed')}")
    print(f"   perfect_fit:         {res.get('perfect_fit')}")
    print(f"   perfect_fit_by_r2:   {res.get('perfect_fit_by_r2')}")
    print(f"   runtime_sec:         {res.get('runtime_sec')}")
    print(f"   error:               {res.get('error')}")
    print("-" * 72)


def safe_numeric_values(results, key):
    out = []
    for r in results:
        value = r.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if np.isfinite(value):
            out.append(value)
    return out


def print_summary(title, results, overall_start):
    total = len(results)
    valid_count = sum(1 for r in results if r.get("valid_formula_found"))
    passed_count = sum(1 for r in results if r.get("passed"))
    perfect_mse_count = sum(1 for r in results if r.get("perfect_fit"))
    perfect_r2_count = sum(1 for r in results if r.get("perfect_fit_by_r2"))
    test_mses = safe_numeric_values(results, "best_test_mse")
    test_r2s = safe_numeric_values(results, "test_r2")
    complexities = safe_numeric_values(results, "expr_complexity")

    print("\n" + "=" * 72)
    print(f"SUMMARY: {title}")
    print("=" * 72)
    print(f"Total Files:            {total}")
    print(f"Valid Formulas Found:   {valid_count} ({(100.0 * valid_count / total) if total else 0:.1f}%)")
    print(f"Calculated Test MSEs:   {len(test_mses)}")
    print("-" * 52)
    print(f"PASSED (test MSE <= {MSE_THRESHOLD}): {passed_count} ({(100.0 * passed_count / total) if total else 0:.1f}%)")
    print(f"Perfect Fits by MSE:    {perfect_mse_count}")
    print(f"Perfect Fits by R2 >= {PERFECT_FIT_R2_THRESHOLD}: {perfect_r2_count}")
    if test_mses:
        print(f"   Mean Test MSE:       {np.mean(test_mses):.6f}")
        print(f"   Median Test MSE:     {np.median(test_mses):.6f}")
    if test_r2s:
        print(f"   Mean Test R2:        {np.mean(test_r2s):.6f}")
        print(f"   Median Test R2:      {np.median(test_r2s):.6f}")
    if complexities:
        print(f"   Mean Complexity:     {np.mean(complexities):.3f}")
        print(f"   Median Complexity:   {np.median(complexities):.3f}")
    print(f"Total Runtime:          {time.time() - overall_start:.2f} seconds")
    print("=" * 72)


def get_argparser():
    parser = argparse.ArgumentParser(description="Run CPU SR baselines on local benchmarks.")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["sldbench", "srbench", "llmsrbench", "srsd"],
        help="Benchmark wrapper to reuse.",
    )
    parser.add_argument("--method", default="gplearn", choices=sorted(METHOD_FITTERS), help="CPU baseline method.")
    parser.add_argument(
        "--config",
        default=None,
        help="Method YAML config. If omitted, a CPU fast config is chosen for the selected method.",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Optional global cap after benchmark filtering.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed passed to gplearn if config omits one.")
    parser.add_argument("--results-root", default=None, help="Output directory. Defaults under CPU_SR_RESULTS_BASE.")
    parser.add_argument("--dry-run", action="store_true", help="Only collect and save selected tasks.")
    parser.add_argument(
        "--isolate-cases",
        action="store_true",
        help="Run each case in a child Python process so hard crashes are recorded per case.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip completed per-case result JSON files.")
    parser.add_argument("--single-case-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-result-json", default=None, help=argparse.SUPPRESS)
    return parser


def default_config_for_method(method: str) -> str:
    external = PROJECT_ROOT / "srsd-benchmark" / "external"
    paths = {
        "gplearn": external / "gplearn" / "configs" / "cpu_fast.yaml",
        "aifeynman": external / "ai_feynman2" / "configs" / "cpu_practical.yaml",
        "bingo": PROJECT_ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "bingo_limited.yaml",
        "deap": external / "deap" / "configs" / "cpu_practical.yaml",
        "dso": PROJECT_ROOT / "evaluation_suites" / "gpu_symbolic_regression_fourbench" / "configs" / "dso_limited.yaml",
        "ffx": PROJECT_ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "ffx_fast.yaml",
        "psrn": PROJECT_ROOT / "evaluation_suites" / "gpu_symbolic_regression_fourbench" / "configs" / "psrn_limited.yaml",
        "pysr": external / "pysr" / "configs" / "cpu_fast.yaml",
        "pyoperon": external / "pyoperon" / "configs" / "cpu_fast.yaml",
        "rils_rols": PROJECT_ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "rils_rols_limited.yaml",
        "itea": external / "itea" / "configs" / "cpu_fast.yaml",
    }
    return str(paths[method])


def main():
    args = get_argparser().parse_args()
    overall_start = time.time()
    results_root = args.results_root or os.path.join(
        DEFAULT_RESULTS_BASE,
        f"cpu_{args.method}_{args.benchmark}_{RUN_TIMESTAMP}",
    )
    os.makedirs(results_root, exist_ok=True)

    selected_csv = os.path.join(results_root, f"{args.benchmark}_selected_tasks.csv")
    results_csv = os.path.join(results_root, f"all_{args.benchmark}_{args.method}_results.csv")
    config_path = args.config or default_config_for_method(args.method)

    print(f"[INFO] CPU baseline method: {args.method}")
    print(f"[INFO] Benchmark:           {args.benchmark}")
    print(f"[INFO] Config:              {config_path}")
    print(f"[INFO] Results root:        {results_root}")

    mod, df_tasks = collect_tasks(args.benchmark)
    if args.max_cases is not None and args.max_cases > 0 and len(df_tasks) > args.max_cases:
        df_tasks = df_tasks.head(args.max_cases).copy()
    if args.single_case_index is None:
        df_tasks.to_csv(selected_csv, index=False, encoding="utf-8-sig")
        print(f"Selection saved to: {selected_csv}")

    if len(df_tasks) == 0:
        print("No tasks selected.")
        return
    if args.dry_run:
        print("Dry run only, exiting after selection.")
        return

    if args.single_case_index is not None:
        run_single_case_child(args, mod, df_tasks, config_path)
        return

    mode = "isolated subprocesses" if args.isolate_cases else "in-process"
    print(f"[INFO] Selected {len(df_tasks)} {args.benchmark} cases. Starting execution ({mode})...", flush=True)

    all_results = []
    total = len(df_tasks)
    for idx, (_, row) in enumerate(df_tasks.iterrows(), start=1):
        row_dict = row.to_dict()
        print(f"[{idx}/{total}] Starting: {row_case_name(args.benchmark, row_dict)}", flush=True)
        if args.isolate_cases:
            res = run_isolated_case(args, idx, total, row_dict, config_path, results_root)
        else:
            case_slug = sanitize_name(row_case_name(args.benchmark, row_dict) or f"case_{idx}")
            case_result_dir = Path(results_root) / "case_results"
            case_result_dir.mkdir(parents=True, exist_ok=True)
            result_json = case_result_dir / f"{idx:04d}_{case_slug}.json"
            if args.resume and result_json.exists():
                with open(result_json, "r", encoding="utf-8") as fp:
                    res = json.load(fp)
                res["resumed_existing_result"] = True
            else:
                res = run_one(
                    benchmark=args.benchmark,
                    method=args.method,
                    mod=mod,
                    row=row_dict,
                    config_path=config_path,
                    random_state=args.random_state,
                )
                res["completed_index"] = int(idx)
                res["total_tasks"] = int(total)
                safe_result = json_safe_record(res)
                tmp_result_json = result_json.with_suffix(result_json.suffix + ".tmp")
                with open(tmp_result_json, "w", encoding="utf-8") as fp:
                    json.dump(safe_result, fp, ensure_ascii=False, allow_nan=False, indent=2)
                os.replace(tmp_result_json, result_json)
        all_results.append(res)
        print_one_result(args.benchmark, idx, total, row_dict, res)
        pd.DataFrame([json_safe_record(r) for r in all_results]).to_csv(results_csv, index=False)

    print(f"Saved global results to: {results_csv}")
    print_summary(f"ALL {args.benchmark.upper()} CASES ({args.method})", all_results, overall_start)


if __name__ == "__main__":
    main()
