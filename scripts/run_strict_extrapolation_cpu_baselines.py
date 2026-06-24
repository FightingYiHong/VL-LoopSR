#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run strict-budget CPU SR baselines on controlled extrapolation splits."""

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

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_cpu_baseline_benchmarks as cpu_sr


class ExtrapolationCase:
    def __init__(
        self,
        benchmark,
        case_name,
        structure_type,
        n_features,
        true_vars,
        fn,
        train_sampler,
        test_sampler,
        train_range_label,
        test_range_label,
        distance_to_train_range=1.0,
    ):
        self.experiment = "extrapolation"
        self.benchmark = benchmark
        self.case_name = case_name
        self.structure_type = structure_type
        self.n_features = n_features
        self.true_vars = true_vars
        self.fn = fn
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.train_range_label = train_range_label
        self.test_range_label = test_range_label
        self.distance_to_train_range = distance_to_train_range


def _sample(n_features, low, high):
    def inner(n, rng):
        return rng.uniform(low, high, size=(n, n_features))
    return inner


def _sample_ranges(ranges):
    ranges = list(ranges)

    def inner(n, rng):
        cols = [rng.uniform(low, high, size=n) for low, high in ranges]
        return np.vstack(cols).T

    return inner


def make_extrapolation_cases():
    train_1d, test_1d = _sample(1, -1.0, 1.0), _sample(1, 1.0, 3.0)
    train_2d, test_2d = _sample(2, -1.0, 1.0), _sample(2, 1.0, 3.0)
    cases = [
        ExtrapolationCase("Feynman", "ood_quadratic_1d", "polynomial", 1, [0], lambda x: 0.5 * x[:, 0] ** 2 - 1.2 * x[:, 0] + 0.7, train_1d, test_1d, "[-1,1]", "[1,3]"),
        ExtrapolationCase("Feynman", "ood_cubic_turning_1d", "polynomial", 1, [0], lambda x: 0.08 * x[:, 0] ** 3 - 0.6 * x[:, 0] ** 2 + 0.4 * x[:, 0] + 1.0, train_1d, test_1d, "[-1,1]", "[1,3]"),
        ExtrapolationCase("Feynman", "ood_exp_growth_1d", "exp_log", 1, [0], lambda x: np.exp(0.3 * x[:, 0]) + 0.25 * x[:, 0], train_1d, test_1d, "[-1,1]", "[1,3]"),
        ExtrapolationCase("SLDBench", "ood_log_scaling_1d", "exp_log", 1, [0], lambda x: np.log(x[:, 0] + 2.2) + 0.1 * x[:, 0], train_1d, test_1d, "[-1,1]", "[1,3]"),
        ExtrapolationCase("SLDBench", "ood_sqrt_scaling_1d", "function_composition", 1, [0], lambda x: 1.5 * np.sqrt(np.abs(x[:, 0]) + 1.2) + 0.2 * x[:, 0], train_1d, test_1d, "[-1,1]", "[1,3]"),
        ExtrapolationCase("selected_1d_2d", "ood_rational_asymmetry_1d", "division", 1, [0], lambda x: (x[:, 0] + 2.0) / (1.0 + 0.4 * x[:, 0] ** 2), train_1d, test_1d, "[-1,1]", "[1,3]"),
        ExtrapolationCase("selected_1d_2d", "ood_periodic_envelope_1d", "trigonometric", 1, [0], lambda x: np.sin(1.2 * x[:, 0]) + 0.12 * x[:, 0] ** 2, train_1d, test_1d, "[-1,1]", "[1,3]"),
        ExtrapolationCase("Feynman", "ood_rational_2d", "division", 2, [0, 1], lambda x: (x[:, 0] + 1.5) / (1.0 + x[:, 1] ** 2), train_2d, test_2d, "[-1,1]^2", "[1,3]^2", 2.0),
        ExtrapolationCase("selected_1d_2d", "ood_interaction_quadrant_2d", "variable_interaction", 2, [0, 1], lambda x: x[:, 0] * x[:, 1] + 0.25 * x[:, 0] ** 2 - 0.15 * x[:, 1], train_2d, test_2d, "[-1,1]^2", "[1,3]^2", 2.0),
        ExtrapolationCase("selected_1d_2d", "ood_nested_trig_rational_2d", "function_composition", 2, [0, 1], lambda x: np.sin(x[:, 0] * x[:, 1]) / (1.0 + x[:, 1] ** 2) + 0.1 * x[:, 0], train_2d, test_2d, "[-1,1]^2", "[1,3]^2", 2.0),
    ]

    one_d_templates = [
        ("poly_quadratic", "polynomial", "Nguyen", lambda x: 0.7 * x[:, 0] ** 2 - 0.4 * x[:, 0] + 1.0),
        ("poly_cubic", "polynomial", "Nguyen", lambda x: 0.12 * x[:, 0] ** 3 - 0.3 * x[:, 0] ** 2 + x[:, 0] + 0.5),
        ("poly_quartic_shallow", "polynomial", "Nguyen-c", lambda x: 0.02 * x[:, 0] ** 4 - 0.1 * x[:, 0] ** 2 + 0.8),
        ("rational_decay", "division", "Feynman", lambda x: (1.0 + 0.5 * x[:, 0]) / (1.0 + x[:, 0] ** 2)),
        ("rational_offset", "division", "Feynman", lambda x: (x[:, 0] + 2.0) / (0.8 + 0.3 * x[:, 0] ** 2)),
        ("exp_linear", "exp_log", "SLDBench", lambda x: np.exp(0.25 * x[:, 0]) + 0.15 * x[:, 0]),
        ("log_linear", "exp_log", "SLDBench", lambda x: np.log(np.abs(x[:, 0]) + 1.8) + 0.2 * x[:, 0]),
        ("sqrt_scaling", "function_composition", "SLDBench", lambda x: 1.2 * np.sqrt(np.abs(x[:, 0]) + 1.0) + 0.15 * x[:, 0]),
        ("sin_trend", "trigonometric", "selected_1d_2d", lambda x: np.sin(1.1 * x[:, 0]) + 0.08 * x[:, 0] ** 2),
        ("cos_exp_envelope", "function_composition", "selected_1d_2d", lambda x: np.cos(0.8 * x[:, 0]) * np.exp(0.08 * x[:, 0])),
        ("inverse_sqrt", "function_composition", "selected_1d_2d", lambda x: 1.0 / np.sqrt(np.abs(x[:, 0]) + 1.2) + 0.1 * x[:, 0]),
        ("tanh_like_saturation", "function_composition", "selected_1d_2d", lambda x: np.tanh(0.8 * x[:, 0]) + 0.05 * x[:, 0] ** 2),
    ]
    one_d_splits = [
        ("right_near", (-1.0, 1.0), (1.0, 3.0), 1.0),
        ("right_far", (-1.0, 1.0), (3.0, 5.0), 3.0),
        ("left_far", (-1.0, 1.0), (-5.0, -3.0), 3.0),
    ]
    for base_name, structure, benchmark, fn in one_d_templates:
        for split_name, train_range, test_range, distance in one_d_splits:
            cases.append(
                ExtrapolationCase(
                    benchmark,
                    f"ood_{base_name}_{split_name}",
                    structure,
                    1,
                    [0],
                    fn,
                    _sample_ranges([train_range]),
                    _sample_ranges([test_range]),
                    f"[{train_range[0]:g},{train_range[1]:g}]",
                    f"[{test_range[0]:g},{test_range[1]:g}]",
                    distance,
                )
            )

    two_d_templates = [
        ("interaction_poly", "variable_interaction", "Feynman", lambda x: x[:, 0] * x[:, 1] + 0.2 * x[:, 0] ** 2 - 0.1 * x[:, 1]),
        ("rational_surface", "division", "Feynman", lambda x: (x[:, 0] + 1.5) / (1.0 + x[:, 1] ** 2)),
        ("nested_trig_ratio", "function_composition", "SRSD", lambda x: np.sin(x[:, 0] * x[:, 1]) / (1.0 + x[:, 1] ** 2) + 0.1 * x[:, 0]),
        ("exp_interaction", "exp_log", "SLDBench", lambda x: np.exp(0.12 * x[:, 0]) + 0.25 * x[:, 0] * x[:, 1]),
        ("log_product", "exp_log", "SLDBench", lambda x: np.log(np.abs(x[:, 0] * x[:, 1]) + 1.5) + 0.1 * x[:, 0]),
        ("sin_separable", "trigonometric", "selected_1d_2d", lambda x: np.sin(x[:, 0]) + 0.5 * np.cos(x[:, 1])),
        ("coupled_quadratic", "variable_interaction", "selected_1d_2d", lambda x: 0.5 * x[:, 0] ** 2 + 0.3 * x[:, 1] ** 2 + 0.2 * x[:, 0] * x[:, 1]),
        ("reciprocal_interaction", "division", "selected_1d_2d", lambda x: (1.0 + x[:, 0] * x[:, 1]) / (1.0 + x[:, 0] ** 2 + 0.5 * x[:, 1] ** 2)),
    ]
    two_d_splits = [
        ("same_quadrant", [(-1.0, 1.0), (-1.0, 1.0)], [(1.0, 3.0), (1.0, 3.0)], 2.0),
        ("opposite_quadrant", [(-1.0, 1.0), (-1.0, 1.0)], [(1.0, 3.0), (-3.0, -1.0)], 2.0),
    ]
    for base_name, structure, benchmark, fn in two_d_templates:
        for split_name, train_ranges, test_ranges, distance in two_d_splits:
            cases.append(
                ExtrapolationCase(
                    benchmark,
                    f"ood_{base_name}_{split_name}",
                    structure,
                    2,
                    [0, 1],
                    fn,
                    _sample_ranges(train_ranges),
                    _sample_ranges(test_ranges),
                    " x ".join(f"[{lo:g},{hi:g}]" for lo, hi in train_ranges),
                    " x ".join(f"[{lo:g},{hi:g}]" for lo, hi in test_ranges),
                    distance,
                )
            )

    return cases


DEFAULT_CONFIGS = {
    "gplearn": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "gplearn_100s_1thread.yaml",
    "pyoperon": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "pyoperon_100s_1thread.yaml",
    "pysr": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "pysr_100s_1thread.yaml",
    "dso": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "dso_100s.yaml",
    "itea": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "itea_100s.yaml",
    "bingo": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "bingo_100s.yaml",
    "ffx": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "ffx_100s.yaml",
    "rils_rols": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "rils_rols_100s.yaml",
    "deap": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "deap_100s.yaml",
    "psrn": ROOT / "evaluation_suites" / "gpu_symbolic_regression_fourbench" / "configs" / "psrn_100s_safe.yaml",
}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def make_split(case_idx: int, repeat_seed: int, n_train: int, n_val: int, n_test: int):
    cases = make_extrapolation_cases()
    case = cases[case_idx]
    rng = np.random.default_rng(20260527 + case_idx * 1009 + repeat_seed * 10007)
    x_train = case.train_sampler(n_train, rng)
    x_val = case.train_sampler(n_val, rng)
    x_test = case.test_sampler(n_test, rng)
    y_train = case.fn(x_train)
    y_val = case.fn(x_val)
    y_test = case.fn(x_test)

    def frame(x, y):
        data = {f"x{i}": x[:, i].astype(float) for i in range(x.shape[1])}
        data["y"] = y.astype(float)
        return pd.DataFrame(data)

    meta = {
        "experiment": case.experiment,
        "benchmark": case.benchmark,
        "case_name": case.case_name,
        "case_index": int(case_idx + 1),
        "structure_type": case.structure_type,
        "n_features": int(case.n_features),
        "true_vars": "|".join(f"x{i}" for i in case.true_vars),
        "train_range": case.train_range_label,
        "test_range": case.test_range_label,
        "distance_to_train_range": float(case.distance_to_train_range),
        "repeat_seed": int(repeat_seed),
    }
    return case, frame(x_train, y_train), frame(x_val, y_val), frame(x_test, y_test), meta


def prepare_budget_config(method: str, config_path: str | Path, budget_sec: float, result_json: str | Path) -> Path:
    source = Path(config_path)
    config = yaml.load(source.read_text(encoding="utf-8"), Loader=yaml.FullLoader) or {}
    inner = max(1.0, float(budget_sec) - 5.0)

    config.setdefault("runtime", {})
    config["runtime"]["max_fit_seconds"] = inner
    if method == "pyoperon":
        config.setdefault("model", {}).setdefault("kwargs", {})
        config["model"]["kwargs"]["max_time"] = int(max(1.0, inner - 1.0))
    elif method == "pysr":
        config.setdefault("fit", {})
        # PySR's Julia backend needs time after the search timeout to flush
        # equations back to Python. Keep the search budget below the parent
        # wall-clock budget so a valid incumbent is not lost at teardown.
        pysr_inner = min(max(8.0, float(budget_sec) * 0.20), max(8.0, float(budget_sec) - 75.0))
        config["runtime"]["max_fit_seconds"] = pysr_inner
        config["fit"]["timeout_in_seconds"] = pysr_inner
        if "ncyclesperiteration" in config["fit"]:
            config["fit"]["ncycles_per_iteration"] = config["fit"].pop("ncyclesperiteration")
        config["fit"]["niterations"] = min(int(config["fit"].get("niterations", 40)), 40)
        config["fit"]["populations"] = min(int(config["fit"].get("populations", 3)), 3)
        config["fit"]["population_size"] = min(int(config["fit"].get("population_size", 40)), 40)
        config["fit"]["ncycles_per_iteration"] = min(int(config["fit"].get("ncycles_per_iteration", 50)), 50)
        config["fit"]["maxsize"] = min(int(config["fit"].get("maxsize", 28)), 28)
        config["fit"].pop("early_stop_condition", None)
        config["fit"]["progress"] = False
        config["fit"]["verbosity"] = 0
    elif method == "dso":
        config.setdefault("runtime", {})
        config["runtime"]["max_fit_seconds"] = inner
    elif method == "psrn":
        config.setdefault("fit", {})
        config["fit"]["time_limit"] = int(min(max(1.0, inner), max(1.0, float(budget_sec) - 30.0)))
        config["fit"]["n_down_sample"] = min(int(config["fit"].get("n_down_sample", 200)), 200)
        config.setdefault("runtime", {})
        config["runtime"]["max_train_rows"] = min(int(config["runtime"].get("max_train_rows", 500)), 500)

    out = Path(result_json).parent / f"_budget_{method}_{int(float(budget_sec))}s.yaml"
    out.write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")
    return out


def run_child(args) -> int:
    source_config = args.config or str(DEFAULT_CONFIGS[args.method])
    config = prepare_budget_config(args.method, source_config, args.case_budget_sec, args.single_result_json)
    _, train_df, val_df, test_df, meta = make_split(
        args.case_index - 1,
        args.repeat_seed,
        args.n_train,
        args.n_val,
        args.n_test,
    )
    started = time.time()
    try:
        if args.method not in cpu_sr.METHOD_FITTERS:
            raise ValueError(f"unsupported method: {args.method}")
        result = cpu_sr.METHOD_FITTERS[args.method](
            train_df,
            val_df,
            test_df,
            config_path=config,
            random_state=args.random_state + args.case_index * 100 + args.repeat_seed,
        )
        result.update(meta)
        result.update(
            {
                "method": args.method,
                "budget_sec": float(args.case_budget_sec),
                "timed_out": False,
                "runtime_sec": float(time.time() - started),
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "train_mse": result.get("best_train_mse"),
                "extrapolation_mse": result.get("best_test_mse"),
                "extrapolation_r2": result.get("test_r2"),
                "config_path": str(config),
                "source_config_path": str(source_config),
                "error": result.get("error"),
            }
        )
    except BaseException as exc:
        timed_out = isinstance(exc, TimeoutError) or "exceeded" in repr(exc).lower() or "timeout" in repr(exc).lower()
        result = {
            **meta,
            "method": args.method,
            "budget_sec": float(args.case_budget_sec),
            "timed_out": bool(timed_out),
            "runtime_sec": float(time.time() - started),
            "valid_formula_found": False,
            "passed": False,
            "best_expr": "",
            "best_test_mse": None,
            "test_r2": None,
            "train_mse": None,
            "extrapolation_mse": None,
            "extrapolation_r2": None,
            "config_path": str(config),
            "source_config_path": str(source_config),
            "error": repr(exc),
        }
    Path(args.single_result_json).write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


def terminate_group(pid: int, grace_sec: float = 2.0):
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        return
    deadline = time.time() + grace_sec
    while time.time() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        pass


def timeout_result(args, case_idx: int, repeat_seed: int, runtime_sec: float, log_path: Path):
    _, _, _, _, meta = make_split(case_idx - 1, repeat_seed, args.n_train, args.n_val, args.n_test)
    return {
        **meta,
        "method": args.method,
        "budget_sec": float(args.case_budget_sec),
        "timed_out": True,
        "runtime_sec": float(runtime_sec),
        "valid_formula_found": False,
        "passed": False,
        "best_expr": "",
        "best_test_mse": None,
        "test_r2": None,
        "train_mse": None,
        "extrapolation_mse": None,
        "extrapolation_r2": None,
        "config_path": str(args.config or DEFAULT_CONFIGS[args.method]),
        "case_log_path": str(log_path),
        "error": f"case exceeded strict budget of {args.case_budget_sec}s",
    }


def summarize(rows: list[dict], out_dir: Path, method: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    for column in [
        "timed_out",
        "passed",
        "train_mse",
        "extrapolation_mse",
        "extrapolation_r2",
        "runtime_sec",
        "expr_complexity",
    ]:
        if column not in df:
            df[column] = np.nan
    df.to_csv(out_dir / f"all_extrapolation_{method}_strict100_results.csv", index=False)
    summary = (
        df.groupby(["benchmark", "method"], dropna=False)
        .agg(
            n=("case_name", "count"),
            timeout_rate=("timed_out", "mean"),
            pass_rate=("passed", "mean"),
            median_train_mse=("train_mse", "median"),
            median_extrapolation_mse=("extrapolation_mse", "median"),
            median_extrapolation_r2=("extrapolation_r2", "median"),
            median_runtime=("runtime_sec", "median"),
            median_complexity=("expr_complexity", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / f"summary_extrapolation_{method}_strict100.csv", index=False)


def run_parent(args) -> int:
    out_dir = Path(args.out_dir)
    log_dir = out_dir / "case_logs" / args.method
    result_dir = out_dir / "case_results" / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    cases = make_extrapolation_cases()
    if args.max_cases:
        cases = cases[: args.max_cases]
    manifest = {
        "method": args.method,
        "case_budget_sec": args.case_budget_sec,
        "repeat_seeds": args.repeat_seeds,
        "n_cases": len(cases),
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "config": str(args.config or DEFAULT_CONFIGS[args.method]),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / f"manifest_{args.method}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict] = []
    total = len(cases) * args.repeat_seeds
    done = 0
    for case_idx, case in enumerate(cases, start=1):
        for repeat_seed in range(args.repeat_seeds):
            done += 1
            slug = f"{case_idx:03d}_{case.case_name}_seed{repeat_seed}"
            result_json = result_dir / f"{slug}.json"
            log_path = log_dir / f"{slug}.log"
            if result_json.exists() and args.resume:
                row = json.loads(result_json.read_text(encoding="utf-8"))
                rows.append(row)
                summarize(rows, out_dir, args.method)
                print(f"[SKIP {done}/{total}] {case.case_name} seed={repeat_seed}", flush=True)
                continue
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--method",
                args.method,
                "--case-index",
                str(case_idx),
                "--repeat-seed",
                str(repeat_seed),
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
            print(f"[RUN {done}/{total}] {args.method} {case.case_name} seed={repeat_seed}", flush=True)
            with open(log_path, "w", encoding="utf-8", errors="replace") as fp:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=fp,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                try:
                    parent_timeout = float(args.case_budget_sec)
                    if args.method == "pysr":
                        parent_timeout += 45.0
                    return_code = proc.wait(timeout=parent_timeout)
                except subprocess.TimeoutExpired:
                    terminate_group(proc.pid)
                    row = timeout_result(args, case_idx, repeat_seed, time.time() - started, log_path)
                    result_json.write_text(json.dumps(json_safe(row), ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    if result_json.exists():
                        row = json.loads(result_json.read_text(encoding="utf-8"))
                    else:
                        row = timeout_result(args, case_idx, repeat_seed, time.time() - started, log_path)
                        row["timed_out"] = False
                        row["error"] = f"child exited rc={return_code} without result json"
                        result_json.write_text(json.dumps(json_safe(row), ensure_ascii=False, indent=2), encoding="utf-8")
                    row["return_code"] = int(return_code)
                    row["case_log_path"] = str(log_path)
                    result_json.write_text(json.dumps(json_safe(row), ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)
            summarize(rows, out_dir, args.method)
            print(
                f"[DONE {done}/{total}] timeout={row.get('timed_out')} mse={row.get('extrapolation_mse')} "
                f"r2={row.get('extrapolation_r2')} sec={row.get('runtime_sec'):.1f}",
                flush=True,
            )
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(DEFAULT_CONFIGS))
    parser.add_argument("--out-dir", required=False)
    parser.add_argument("--config", default=None)
    parser.add_argument("--case-budget-sec", type=float, default=100.0)
    parser.add_argument("--repeat-seeds", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-val", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case-index", type=int, default=1)
    parser.add_argument("--repeat-seed", type=int, default=0)
    parser.add_argument("--single-result-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.child:
        if not args.single_result_json:
            raise ValueError("--single-result-json is required in child mode")
        return run_child(args)
    if not args.out_dir:
        raise ValueError("--out-dir is required")
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
