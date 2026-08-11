#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate fixed noisy symbolic-regression replicas for robustness experiments."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FormulaSpec:
    formula_id: str
    benchmark: str
    structure_type: str
    expression: str
    variables: list[str]
    sample_low: float = -2.0
    sample_high: float = 2.0


def sanitize_name(text: str) -> str:
    text = str(text).replace("/", "__").replace("\\", "__")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "case"


def safe_eval_expression(expr: str, df: pd.DataFrame, variables: list[str]) -> np.ndarray:
    local_vars = {name: df[name].to_numpy(dtype=float) for name in variables}
    local_vars.update(
        {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "tanh": np.tanh,
            "exp": lambda x: np.exp(np.clip(x, -30, 30)),
            "log": lambda x: np.log(np.abs(x) + 1.0e-12),
            "sqrt": lambda x: np.sqrt(np.abs(x)),
            "abs": np.abs,
            "Abs": np.abs,
            "pi": np.pi,
            "E": np.e,
        }
    )
    with np.errstate(all="ignore"):
        y = eval(str(expr).replace("^", "**"), {"__builtins__": {}}, local_vars)
    y = np.asarray(y, dtype=float)
    if y.ndim == 0:
        y = np.full(len(df), float(y), dtype=float)
    y = y.reshape(-1)
    if len(y) != len(df) or not np.all(np.isfinite(y)):
        raise ValueError(f"non-finite or wrong-length expression output for {expr}")
    return y


def formula_specs() -> list[FormulaSpec]:
    specs = [
        FormulaSpec("nguyen_poly", "Nguyen", "polynomial", "x1**3 + x1**2 + x1", ["x1"]),
        FormulaSpec("nguyen_rational", "Nguyen", "rational", "x1/(1 + x2**2)", ["x1", "x2"]),
        FormulaSpec("feynman_trig", "Feynman", "trigonometric", "sin(x1) + 0.5*cos(x2)", ["x1", "x2"]),
        FormulaSpec("feynman_interaction", "Feynman", "interaction", "x1*x2 + 0.25*x3**2", ["x1", "x2", "x3"]),
        FormulaSpec("srsd_exp_log", "SRSD", "exp_log", "exp(0.3*x1) + log(abs(x2) + 2.0)", ["x1", "x2"]),
        FormulaSpec("srsd_nested", "SRSD", "function_composition", "sin(x1*x2)/(1 + x3**2)", ["x1", "x2", "x3"]),
        FormulaSpec("realistic_decay", "Realistic", "exp_trig", "exp(-0.25*abs(x1))*sin(x2) + 0.15*x3", ["x1", "x2", "x3"]),
        FormulaSpec("realistic_sensor", "Realistic", "mixed_smooth", "log(abs(x1) + 2.0) + 0.35*x2**2 - 0.2*cos(x3)", ["x1", "x2", "x3"]),
    ]
    templates = [
        ("poly_quad_shift", "Nguyen", "polynomial", "x1**2 + 0.5*x2 - 0.25"),
        ("poly_cubic_mix", "Nguyen", "polynomial", "0.5*x1**3 - x2**2 + x1*x2"),
        ("rational_offset", "Nguyen", "rational", "(x1 + x2)/(1 + abs(x1))"),
        ("rational_quad", "Nguyen", "rational", "(x1**2 + 0.25*x2)/(1 + x3**2)"),
        ("trig_single", "Feynman", "trigonometric", "sin(1.2*x1) + 0.1*x1"),
        ("trig_product", "Feynman", "trigonometric_interaction", "sin(x1)*cos(x2) + 0.2*x3"),
        ("exp_decay", "Feynman", "exponential", "exp(-0.4*abs(x1)) + 0.3*x2"),
        ("log_poly", "SRSD", "log_polynomial", "log(abs(x1) + 1.5) + 0.2*x2**2"),
        ("nested_trig_poly", "SRSD", "function_composition", "sin(x1 + 0.5*x2**2)"),
        ("nested_rational_trig", "SRSD", "function_composition", "sin(x1)/(1 + x2**2) + 0.1*x3"),
        ("interaction_three", "Realistic", "interaction", "x1*x2 - 0.4*x2*x3 + 0.1*x1"),
        ("mixed_smooth", "Realistic", "mixed_smooth", "sqrt(abs(x1) + 1.0) + 0.25*cos(x2) - 0.1*x3"),
        ("exp_log_mix", "Realistic", "exp_log", "exp(0.15*x1) + log(abs(x2) + 1.0) - 0.2*x3"),
        ("envelope_periodic", "Realistic", "exp_trig", "exp(-0.2*x1**2)*sin(1.5*x2) + 0.1*x3"),
    ]
    suffixes = [
        ("a", {"x1": "x1", "x2": "x2", "x3": "x3"}),
        ("b", {"x1": "x2", "x2": "x3", "x3": "x1"}),
        ("c", {"x1": "x3", "x2": "x1", "x3": "x2"}),
    ]
    for base_name, bench, family, expr in templates:
        for suffix, mapping in suffixes:
            if len(specs) >= 50:
                break
            mapped = expr
            for old, new in sorted(mapping.items(), reverse=True):
                mapped = re.sub(rf"\b{old}\b", new, mapped)
            vars_used = sorted(set(re.findall(r"\bx[1-5]\b", mapped)), key=lambda x: int(x[1:]))
            specs.append(FormulaSpec(f"{base_name}_{suffix}", bench, family, mapped, vars_used))
        if len(specs) >= 50:
            break
    return specs[:50]


def make_features(spec: FormulaSpec, n: int, rng: np.random.Generator) -> pd.DataFrame:
    arr = rng.uniform(float(spec.sample_low), float(spec.sample_high), size=(n, len(spec.variables)))
    return pd.DataFrame(arr, columns=spec.variables)


def add_relative_noise(y: np.ndarray, level: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    sigma = float(np.std(y)) * float(level)
    if not math.isfinite(sigma) or sigma <= 0:
        return np.asarray(y, dtype=float).copy(), 0.0
    return np.asarray(y, dtype=float) + rng.normal(0.0, sigma, size=len(y)), sigma


def write_split(path: Path, df: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def generate(args) -> None:
    out_dir = Path(args.out_dir)
    split_root = out_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = [float(x) for x in str(args.noise_levels).replace(" ", ",").split(",") if x.strip()]
    specs = formula_specs()[: int(args.formula_count)]
    formulas = [asdict(spec) | {"variables": "|".join(spec.variables)} for spec in specs]
    pd.DataFrame(formulas).to_csv(out_dir / "formulas.csv", index=False)

    manifest_rows = []
    case_index = 1
    for spec in specs:
        for level in levels:
            for repeat_seed in range(int(args.repeat_seeds)):
                seed = int(args.random_state) + repeat_seed * 1_000_000 + case_index * 10_000
                train_rng = np.random.default_rng(seed + 1)
                val_rng = np.random.default_rng(seed + 2)
                test_rng = np.random.default_rng(seed + 3)
                noise_rng = np.random.default_rng(seed + 4)
                train_x = make_features(spec, int(args.n_train), train_rng)
                val_x = make_features(spec, int(args.n_val), val_rng)
                test_x = make_features(spec, int(args.n_test), test_rng)
                train_clean_y = safe_eval_expression(spec.expression, train_x, spec.variables)
                val_clean_y = safe_eval_expression(spec.expression, val_x, spec.variables)
                test_clean_y = safe_eval_expression(spec.expression, test_x, spec.variables)
                train_noisy_y, train_sigma = add_relative_noise(train_clean_y, level, noise_rng)
                val_noisy_y, val_sigma = add_relative_noise(val_clean_y, level, noise_rng)

                case_id = f"{sanitize_name(spec.formula_id)}__noise_{level:g}__seed_{repeat_seed}"
                case_dir = split_root / case_id
                train_noisy = train_x.assign(y=train_noisy_y)
                val_noisy = val_x.assign(y=val_noisy_y)
                test_clean = test_x.assign(y=test_clean_y)
                train_clean = train_x.assign(y=train_clean_y)
                val_clean = val_x.assign(y=val_clean_y)
                row = {
                    "case_index": case_index,
                    "case_id": case_id,
                    "formula_id": spec.formula_id,
                    "benchmark": spec.benchmark,
                    "case_name": f"{spec.formula_id}_noise_{level:g}",
                    "base_case_name": spec.formula_id,
                    "structure_type": spec.structure_type,
                    "expression": spec.expression,
                    "variables": "|".join(spec.variables),
                    "true_variables": "|".join(spec.variables),
                    "noise_level": level,
                    "noise_kind": "relative_gaussian_y_train_val",
                    "repeat_seed": repeat_seed,
                    "n_train": int(args.n_train),
                    "n_val": int(args.n_val),
                    "n_test": int(args.n_test),
                    "train_noise_sigma": train_sigma,
                    "val_noise_sigma": val_sigma,
                    "train_noisy_path": write_split(case_dir / "train_noisy.csv", train_noisy),
                    "val_noisy_path": write_split(case_dir / "val_noisy.csv", val_noisy),
                    "test_clean_path": write_split(case_dir / "test_clean.csv", test_clean),
                    "train_clean_path": write_split(case_dir / "train_clean.csv", train_clean),
                    "val_clean_path": write_split(case_dir / "val_clean.csv", val_clean),
                    "split_dir": str(case_dir),
                }
                (case_dir / "meta.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                manifest_rows.append(row)
                case_index += 1
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    write_noise_txt(out_dir, args, len(specs), len(levels), len(manifest_rows))


def write_noise_txt(out_dir: Path, args, n_formulas: int, n_levels: int, n_tasks: int) -> None:
    text = f"""NoiseRobust-SR v1 construction log

Purpose
  Fixed noisy symbolic-regression replicas for evaluating structural recovery
  under observation noise. The method sees noisy train/validation targets but
  is scored against clean test targets and known symbolic skeletons.

Construction
  1. Select {n_formulas} analytic formula specifications with ground-truth expressions.
  2. For each formula, sample variables uniformly from [-2, 2].
  3. Generate clean targets y_clean from the analytic expression.
  4. For train and validation only, add target-scale Gaussian noise:
       y_noisy = y_clean + Normal(0, alpha * std(y_clean)).
  5. Keep test targets clean for final structural recovery scoring.
  6. Repeat each formula/noise-level pair for {args.repeat_seeds} independent seeds.

Noise levels
  {args.noise_levels}

Dataset size
  formulas: {n_formulas}
  noise levels per formula: {n_levels}
  seeds per formula/noise pair: {args.repeat_seeds}
  total tasks: {n_tasks}
  train rows per task: {args.n_train}
  validation rows per task: {args.n_val}
  clean test rows per task: {args.n_test}
  total rows across all task files: {n_tasks * (int(args.n_train) + int(args.n_val) + int(args.n_test))}

Files
  manifest.csv: one row per fixed task with paths and noise metadata.
  formulas.csv: one row per ground-truth formula.
  splits/<case_id>/train_noisy.csv: training split used by methods.
  splits/<case_id>/val_noisy.csv: validation split used by methods.
  splits/<case_id>/test_clean.csv: clean scoring split.
  splits/<case_id>/train_clean.csv and val_clean.csv: analysis-only clean labels.
  splits/<case_id>/meta.json: per-task metadata.

Scoring protocol
  Exact Recovery and Skeleton Recovery compare discovered expressions against
  the clean formula and clean test set. No method is given train_clean.csv or
  val_clean.csv during fitting.

Reproducibility
  generator: scripts/generate_noise_robustness_dataset.py
  base random_state: {args.random_state}
"""
    (out_dir / "noise.txt").write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "data" / "generated" / "noise_sr_20"),
    )
    parser.add_argument("--formula-count", type=int, default=20)
    parser.add_argument("--noise-levels", default="0,0.001,0.01")
    parser.add_argument("--repeat-seeds", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
