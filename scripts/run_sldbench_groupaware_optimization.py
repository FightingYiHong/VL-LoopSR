#!/usr/bin/env python3
"""Exploratory, group-aware SLDBench optimization for VEGA-SR.

The experiment follows the public SLDBench task unit:

* one fitted law per benchmark task;
* one shared functional family with group-specific coefficients where needed;
* model/family selection on an extrapolation-style split made only from train;
* one final evaluation on the untouched public test split;
* task-level R2 = 1 - mean(per-output NMSE), matching the public evaluator.

This is an exploratory specialization experiment, not a confirmatory hidden-test
evaluation.  The public task descriptions motivate the candidate families, but
all numerical coefficients are re-fit from the provided training split.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "sldbench_repo"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "paper_experiments"
    / "01_standard_recovery"
    / "sldbench_groupaware_optimization"
)

TASK_SCHEMA = {
    "data_constrained_scaling_law": {
        "features": ["unique_tokens", "params", "tokens"],
        "targets": ["loss"],
    },
    "domain_mixture_scaling_law": {
        "features": [f"proportion_domain_{i}" for i in range(1, 6)],
        "targets": [f"loss_domain_{i}" for i in range(1, 6)],
    },
    "easy_question_scaling_law": {
        "features": ["log_flops"],
        "targets": ["brier_score"],
    },
    "lr_bsz_scaling_law": {
        "features": ["lr", "bsz", "data_size", "non_embedding_param_size"],
        "targets": ["lm_loss"],
    },
    "moe_scaling_law": {
        "features": ["num_experts", "dense_parameter_count"],
        "targets": ["loss_validation"],
    },
    "parallel_scaling_law": {
        "features": ["num_params", "parallel_size"],
        "targets": ["loss"],
    },
    "sft_scaling_law": {
        "features": ["sft_data_size"],
        "targets": ["sft_loss"],
    },
    "vocab_scaling_law": {
        "features": ["non_vocab_parameters", "vocab_size", "num_characters"],
        "targets": ["unigram_normalized_loss"],
    },
}

# Public human/oracle reference values reported by the Harbor SLDBench adapter.
# They are comparison metadata only and never enter candidate selection.
HUMAN_REFERENCE_R2 = {
    "vocab_scaling_law": 0.966,
    "sft_scaling_law": 0.957,
    "domain_mixture_scaling_law": 0.671,
    "moe_scaling_law": 0.703,
    "data_constrained_scaling_law": 0.911,
    "lr_bsz_scaling_law": -0.076,
    "parallel_scaling_law": 1.000,
    "easy_question_scaling_law": -1.000,
}


@dataclass
class CandidateResult:
    name: str
    complexity: int
    val_nmse: float
    val_r2: float
    stability_penalty: float
    selection_score: float
    fit_error: str | None = None


@dataclass
class FittedLaw:
    name: str
    complexity: int
    description: str
    predict: Callable[[pd.DataFrame], np.ndarray]
    parameters: dict[str, Any]


def finite_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def task_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    true = finite_array(y_true)
    pred = finite_array(y_pred)
    if true.ndim == 1:
        true = true[:, None]
    if pred.ndim == 1:
        pred = pred[:, None]
    if true.shape != pred.shape or not np.isfinite(pred).all():
        return {
            "nmse": 100000.0,
            "nmae": 100000.0,
            "r2": -99999.0,
            "clipped_r2": -1.0,
            "per_output_nmse": [100000.0] * true.shape[1],
        }
    mse = np.mean((true - pred) ** 2, axis=0)
    mae = np.mean(np.abs(true - pred), axis=0)
    variance = np.var(true, axis=0)
    mad = np.mean(np.abs(true - np.mean(true, axis=0)), axis=0)
    nmse = np.divide(
        mse,
        variance,
        out=np.full_like(mse, 100000.0),
        where=variance > 1e-9,
    )
    nmae = np.divide(
        mae,
        mad,
        out=np.full_like(mae, 100000.0),
        where=mad > 1e-9,
    )
    aggregate_nmse = float(np.mean(nmse))
    r2 = 1.0 - aggregate_nmse
    return {
        "nmse": aggregate_nmse,
        "nmae": float(np.mean(nmae)),
        "r2": float(r2),
        "clipped_r2": float(np.clip(r2, -1.0, 1.0)),
        "per_output_nmse": [float(x) for x in nmse],
    }


def task_arrays(frame: pd.DataFrame, task_name: str) -> tuple[np.ndarray, np.ndarray]:
    schema = TASK_SCHEMA[task_name]
    return (
        frame[schema["features"]].to_numpy(dtype=float),
        frame[schema["targets"]].to_numpy(dtype=float),
    )


def extrapolation_score(frame: pd.DataFrame, task_name: str) -> np.ndarray:
    """Score training rows by distance toward larger/unseen regimes."""
    f = TASK_SCHEMA[task_name]["features"]
    values = frame[f].to_numpy(dtype=float)
    logged = np.sign(values) * np.log1p(np.abs(values))
    lo = np.nanmin(logged, axis=0)
    hi = np.nanmax(logged, axis=0)
    scaled = (logged - lo) / np.maximum(hi - lo, 1e-12)
    weights = np.ones(len(f), dtype=float)
    if task_name == "parallel_scaling_law":
        weights = np.array([0.2, 1.0])
    elif task_name == "vocab_scaling_law":
        weights = np.array([0.2, 1.0, 0.5])
    elif task_name == "moe_scaling_law":
        weights = np.array([0.2, 1.0])
    elif task_name == "data_constrained_scaling_law":
        weights = np.array([0.4, 0.7, 1.0])
    elif task_name == "lr_bsz_scaling_law":
        weights = np.array([0.0, 0.2, 1.0, 1.0])
    return scaled @ weights


def make_train_validation_split(
    frame: pd.DataFrame,
    task_name: str,
    val_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts_train: list[pd.DataFrame] = []
    parts_val: list[pd.DataFrame] = []
    for _, group in frame.groupby("group", sort=False):
        group = group.copy()
        score = extrapolation_score(group, task_name)
        order = np.argsort(score, kind="mergesort")
        n_val = max(1, int(math.ceil(len(group) * val_fraction)))
        if len(group) - n_val < max(4, len(TASK_SCHEMA[task_name]["features"]) + 1):
            n_val = max(1, len(group) // 5)
        val_idx = order[-n_val:]
        train_idx = order[:-n_val]
        parts_train.append(group.iloc[train_idx])
        parts_val.append(group.iloc[val_idx])
    return (
        pd.concat(parts_train, ignore_index=True),
        pd.concat(parts_val, ignore_index=True),
    )


def stable_selection_score(
    law: FittedLaw,
    fit_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    task_name: str,
) -> tuple[float, float, float]:
    _, y_val = task_arrays(val_frame, task_name)
    pred = law.predict(val_frame)
    metrics = task_metrics(y_val, pred)
    val_nmse = float(metrics["nmse"])

    # Internal validity penalty only: reject numerical blow-ups relative to the
    # observed training target range.  It is not an evaluation metric.
    targets = TASK_SCHEMA[task_name]["targets"]
    y_fit = fit_frame[targets].to_numpy(dtype=float)
    scale = np.maximum(np.ptp(y_fit, axis=0), 1e-6)
    centre = np.median(y_fit, axis=0)
    pred2 = pred if pred.ndim == 2 else pred[:, None]
    excess = np.maximum(np.abs(pred2 - centre) / scale - 20.0, 0.0)
    stability_penalty = float(np.mean(excess))
    complexity_tiebreak = 1e-7 * float(law.complexity)
    score = val_nmse + min(stability_penalty, 1e6) + complexity_tiebreak
    return val_nmse, stability_penalty, score


def fit_linear_given_basis(
    basis: np.ndarray,
    y: np.ndarray,
    ridge: float = 1e-10,
) -> np.ndarray:
    x = np.asarray(basis, dtype=float)
    yy = np.asarray(y, dtype=float)
    reg = ridge * np.eye(x.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ yy)


def fit_sft(frame: pd.DataFrame, variant: str) -> FittedLaw:
    params: dict[str, Any] = {}
    alpha_grid = np.geomspace(0.03, 1.5, 48)
    b_grid = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]
    for group_name, group in frame.groupby("group", sort=False):
        d = group["sft_data_size"].to_numpy(dtype=float)
        y = group["sft_loss"].to_numpy(dtype=float)
        d_scale = float(np.median(d))
        best: tuple[float, dict[str, float]] | None = None
        for alpha in alpha_grid:
            powered = np.maximum(d / d_scale, 1e-12) ** alpha
            local_b = b_grid if variant == "saturating_inverse_power" else [0.0]
            for b in local_b:
                z = 1.0 / (powered + b)
                beta = fit_linear_given_basis(np.column_stack([np.ones(len(d)), z]), y)
                pred = beta[0] + beta[1] * z
                mse = float(np.mean((y - pred) ** 2))
                item = {
                    "C": float(beta[0]),
                    "A": float(beta[1]),
                    "alpha": float(alpha),
                    "B": float(b),
                    "D_scale": d_scale,
                }
                if best is None or mse < best[0]:
                    best = (mse, item)
        assert best is not None
        params[str(group_name)] = best[1]

    def predict(df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df), dtype=float)
        for group_name, indices in df.groupby("group", sort=False).groups.items():
            p = params[str(group_name)]
            d = df.loc[indices, "sft_data_size"].to_numpy(dtype=float)
            z = 1.0 / (
                np.maximum(d / p["D_scale"], 1e-12) ** p["alpha"] + p["B"]
            )
            out[df.index.get_indexer(indices)] = p["C"] + p["A"] * z
        return out[:, None]

    return FittedLaw(
        name=variant,
        complexity=4,
        description="group-specific C + A / ((D/D0)^alpha + B)",
        predict=predict,
        parameters=params,
    )


def fit_parallel(frame: pd.DataFrame, variant: str) -> FittedLaw:
    params: dict[str, Any] = {}
    alpha_grid = np.linspace(0.04, 0.6, 38)
    k_grid = np.geomspace(0.01, 4.0, 45)
    beta_grid = np.linspace(0.02, 0.8, 30)
    for group_name, group in frame.groupby("group", sort=False):
        n = group["num_params"].to_numpy(dtype=float)
        psize = group["parallel_size"].to_numpy(dtype=float)
        y = group["loss"].to_numpy(dtype=float)
        n0 = float(np.median(n))
        best: tuple[float, dict[str, float]] | None = None
        if variant == "log_parallel_inverse_power":
            iterator = ((a, k, None) for a in alpha_grid for k in k_grid)
        else:
            iterator = ((a, None, b) for a in alpha_grid for b in beta_grid)
        for alpha, k, beta_exp in iterator:
            if k is not None:
                denom = (n / n0) * (1.0 + k * np.log(np.maximum(psize, 1.0)))
                z = np.maximum(denom, 1e-12) ** (-alpha)
            else:
                z = np.maximum(n / n0, 1e-12) ** (-alpha) * np.maximum(
                    psize, 1.0
                ) ** (-float(beta_exp))
            coef = fit_linear_given_basis(np.column_stack([np.ones(len(n)), z]), y)
            pred = coef[0] + coef[1] * z
            mse = float(np.mean((y - pred) ** 2))
            item = {
                "E": float(coef[0]),
                "A": float(coef[1]),
                "alpha": float(alpha),
                "k": None if k is None else float(k),
                "beta": None if beta_exp is None else float(beta_exp),
                "N_scale": n0,
            }
            if best is None or mse < best[0]:
                best = (mse, item)
        assert best is not None
        params[str(group_name)] = best[1]

    def predict(df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df), dtype=float)
        for group_name, indices in df.groupby("group", sort=False).groups.items():
            q = params[str(group_name)]
            n = df.loc[indices, "num_params"].to_numpy(dtype=float)
            ps = df.loc[indices, "parallel_size"].to_numpy(dtype=float)
            if q["k"] is not None:
                den = (n / q["N_scale"]) * (
                    1.0 + q["k"] * np.log(np.maximum(ps, 1.0))
                )
                z = np.maximum(den, 1e-12) ** (-q["alpha"])
            else:
                z = np.maximum(n / q["N_scale"], 1e-12) ** (-q["alpha"])
                z *= np.maximum(ps, 1.0) ** (-q["beta"])
            out[df.index.get_indexer(indices)] = q["E"] + q["A"] * z
        return out[:, None]

    return FittedLaw(
        name=variant,
        complexity=5,
        description="group-specific asymptotic inverse-power law in model and parallel scale",
        predict=predict,
        parameters=params,
    )


def fit_easy_question(frame: pd.DataFrame, degree: int, alpha: float) -> FittedLaw:
    params: dict[str, Any] = {}
    for group_name, group in frame.groupby("group", sort=False):
        x = group["log_flops"].to_numpy(dtype=float)
        y = group["brier_score"].to_numpy(dtype=float)
        centre = float(np.median(x))
        scale = float(max(np.std(x), 1e-6))
        z = ((x - centre) / scale)[:, None]
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        design = poly.fit_transform(z)
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(design, y)
        params[str(group_name)] = {
            "centre": centre,
            "scale": scale,
            "coef": model.coef_.tolist(),
            "degree": degree,
        }

    def predict(df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df), dtype=float)
        for group_name, indices in df.groupby("group", sort=False).groups.items():
            p = params[str(group_name)]
            x = df.loc[indices, "log_flops"].to_numpy(dtype=float)
            z = (x - p["centre"]) / p["scale"]
            design = np.column_stack([z**power for power in range(degree + 1)])
            out[df.index.get_indexer(indices)] = design @ np.asarray(p["coef"])
        return out[:, None]

    return FittedLaw(
        name=f"polynomial_degree_{degree}_ridge_{alpha:g}",
        complexity=degree + 1,
        description=f"group-specific degree-{degree} polynomial in standardized log FLOPs",
        predict=predict,
        parameters=params,
    )


def fit_domain_mixture(
    frame: pd.DataFrame,
    degree: int,
    alpha: float,
) -> FittedLaw:
    features = TASK_SCHEMA["domain_mixture_scaling_law"]["features"]
    targets = TASK_SCHEMA["domain_mixture_scaling_law"]["targets"]
    params: dict[str, Any] = {}
    for group_name, group in frame.groupby("group", sort=False):
        x = group[features].to_numpy(dtype=float)
        y = group[targets].to_numpy(dtype=float)
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        design = poly.fit_transform(x)
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(design, y)
        params[str(group_name)] = {
            "coef": model.coef_.tolist(),
            "powers": poly.powers_.tolist(),
        }

    def predict(df: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(df), len(targets)), dtype=float)
        for group_name, indices in df.groupby("group", sort=False).groups.items():
            p = params[str(group_name)]
            x = df.loc[indices, features].to_numpy(dtype=float)
            powers = np.asarray(p["powers"], dtype=int)
            design = np.prod(x[:, None, :] ** powers[None, :, :], axis=2)
            pred = design @ np.asarray(p["coef"], dtype=float).T
            out[df.index.get_indexer(indices), :] = pred
        return out

    n_terms = len(next(iter(params.values()))["powers"])
    return FittedLaw(
        name=f"simplex_polynomial_degree_{degree}_ridge_{alpha:g}",
        complexity=n_terms,
        description=f"group-specific degree-{degree} polynomial on mixture proportions",
        predict=predict,
        parameters=params,
    )


def fit_domain_mixture_exponential(frame: pd.DataFrame) -> FittedLaw:
    """Fit c_j + exp(a_j + r @ t_j) for each group and output."""
    features = TASK_SCHEMA["domain_mixture_scaling_law"]["features"]
    targets = TASK_SCHEMA["domain_mixture_scaling_law"]["targets"]
    params: dict[str, Any] = {}
    for group_name, group in frame.groupby("group", sort=False):
        x = group[features].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        group_params: dict[str, Any] = {}
        for target in targets:
            y = group[target].to_numpy(dtype=float)
            y_range = max(float(np.ptp(y)), 1e-4)
            c_candidates = [
                float(np.min(y) - factor * y_range)
                for factor in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
            ]
            best = None
            for c in c_candidates:
                shifted = y - c
                if np.any(shifted <= 0):
                    continue
                beta = fit_linear_given_basis(design, np.log(shifted), ridge=1e-6)
                pred = c + np.exp(np.clip(design @ beta, -30.0, 30.0))
                mse = float(np.mean((y - pred) ** 2))
                if best is None or mse < best[0]:
                    best = (mse, c, beta)
            if best is None:
                raise RuntimeError(f"No exponential mixture fit for {group_name}/{target}")
            group_params[target] = {
                "c": float(best[1]),
                "beta": best[2].tolist(),
            }
        params[str(group_name)] = group_params

    def predict(df: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(df), len(targets)), dtype=float)
        for group_name, indices in df.groupby("group", sort=False).groups.items():
            x = df.loc[indices, features].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(x)), x])
            group_params = params[str(group_name)]
            for j, target in enumerate(targets):
                p = group_params[target]
                out[df.index.get_indexer(indices), j] = p["c"] + np.exp(
                    np.clip(design @ np.asarray(p["beta"], dtype=float), -30.0, 30.0)
                )
        return out

    return FittedLaw(
        name="mixture_exponential",
        complexity=7,
        description="group-specific c_j + exp(a_j + mixture @ t_j)",
        predict=predict,
        parameters=params,
    )


def fit_inverse_power_additive(
    frame: pd.DataFrame,
    task_name: str,
    interaction: bool,
    seed: int,
) -> FittedLaw:
    features = TASK_SCHEMA[task_name]["features"]
    target = TASK_SCHEMA[task_name]["targets"][0]
    x = frame[features].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    scales = np.median(np.maximum(x, 1e-12), axis=0)
    z = np.maximum(x / scales, 1e-12)
    n_features = z.shape[1]

    def unpack(theta: np.ndarray) -> np.ndarray:
        intercept = theta[0]
        coeff = theta[1 : 1 + n_features]
        exponent = np.exp(theta[1 + n_features : 1 + 2 * n_features])
        pred = intercept + np.sum(coeff[None, :] * z ** (-exponent[None, :]), axis=1)
        if interaction:
            pred += theta[-2] * np.prod(z, axis=1) ** (-np.exp(theta[-1]))
        return pred

    p_count = 1 + 2 * n_features + (2 if interaction else 0)
    rng = np.random.default_rng(seed)
    starts = []
    y_range = max(float(np.ptp(y)), 1e-3)
    for _ in range(12):
        theta = np.zeros(p_count)
        theta[0] = np.quantile(y, 0.1) + rng.normal(0, 0.1 * y_range)
        theta[1 : 1 + n_features] = rng.normal(0, y_range, n_features)
        theta[1 + n_features : 1 + 2 * n_features] = rng.uniform(
            np.log(0.03), np.log(1.2), n_features
        )
        if interaction:
            theta[-2] = rng.normal(0, y_range)
            theta[-1] = rng.uniform(np.log(0.03), np.log(1.2))
        starts.append(theta)
    best = None
    for start in starts:
        result = least_squares(
            lambda theta: unpack(theta) - y,
            start,
            loss="soft_l1",
            f_scale=max(float(np.std(y)) * 0.1, 1e-4),
            max_nfev=5000,
        )
        mse = float(np.mean((unpack(result.x) - y) ** 2))
        if best is None or mse < best[0]:
            best = (mse, result.x)
    assert best is not None
    theta = best[1]

    def predict(df: pd.DataFrame) -> np.ndarray:
        xx = df[features].to_numpy(dtype=float)
        zz = np.maximum(xx / scales, 1e-12)
        intercept = theta[0]
        coeff = theta[1 : 1 + n_features]
        exponent = np.exp(theta[1 + n_features : 1 + 2 * n_features])
        pred = intercept + np.sum(
            coeff[None, :] * zz ** (-exponent[None, :]), axis=1
        )
        if interaction:
            pred += theta[-2] * np.prod(zz, axis=1) ** (-np.exp(theta[-1]))
        return pred[:, None]

    return FittedLaw(
        name="additive_inverse_power_interaction"
        if interaction
        else "additive_inverse_power",
        complexity=p_count,
        description="asymptote plus additive inverse powers"
        + (" and one multiplicative interaction" if interaction else ""),
        predict=predict,
        parameters={
            "feature_scales": dict(zip(features, scales.tolist())),
            "theta": theta.tolist(),
        },
    )


def fit_moe(frame: pd.DataFrame, saturation: float | None) -> FittedLaw:
    e = np.maximum(frame["num_experts"].to_numpy(dtype=float), 1.0)
    n = np.maximum(frame["dense_parameter_count"].to_numpy(dtype=float), 1.0)
    y = np.maximum(frame["loss_validation"].to_numpy(dtype=float), 1e-12)
    logn = np.log(n / np.median(n))
    if saturation is None:
        loge = np.log(e)
        name = "log_bilinear"
    else:
        loge = np.log1p(e / saturation)
        name = f"log_bilinear_saturation_{saturation:g}"
    design = np.column_stack([np.ones(len(e)), logn, loge, logn * loge])
    beta = fit_linear_given_basis(design, np.log(y), ridge=1e-8)
    n_scale = float(np.median(n))

    def predict(df: pd.DataFrame) -> np.ndarray:
        ee = np.maximum(df["num_experts"].to_numpy(dtype=float), 1.0)
        nn = np.maximum(df["dense_parameter_count"].to_numpy(dtype=float), 1.0)
        ln = np.log(nn / n_scale)
        le = np.log(ee) if saturation is None else np.log1p(ee / saturation)
        xx = np.column_stack([np.ones(len(ee)), ln, le, ln * le])
        return np.exp(xx @ beta)[:, None]

    return FittedLaw(
        name=name,
        complexity=4,
        description="log loss bilinear in log model size and transformed expert count",
        predict=predict,
        parameters={
            "beta": beta.tolist(),
            "N_scale": n_scale,
            "expert_saturation": saturation,
        },
    )


def fit_lr_bsz(frame: pd.DataFrame, variant: str, seed: int) -> FittedLaw:
    y = frame["lm_loss"].to_numpy(dtype=float)
    lr = np.maximum(frame["lr"].to_numpy(dtype=float), 1e-12)
    bsz = np.maximum(frame["bsz"].to_numpy(dtype=float), 1.0)
    data = np.maximum(frame["data_size"].to_numpy(dtype=float), 1.0)
    params = np.maximum(
        frame["non_embedding_param_size"].to_numpy(dtype=float), 1.0
    )
    scales = {
        "lr": float(np.median(lr)),
        "bsz": float(np.median(bsz)),
        "data": float(np.median(data)),
        "params": float(np.median(params)),
    }
    lrs = lr / scales["lr"]
    bs = bsz / scales["bsz"]
    ds = data / scales["data"]
    ns = params / scales["params"]

    if variant == "hierarchical_optimum":
        p_count = 12

        def model(theta: np.ndarray, a: tuple[np.ndarray, ...]) -> np.ndarray:
            lr0, b0, d0, n0 = a
            A, la, B, lb, C, Q, R, F, gamma, zeta, G, eta = theta
            alpha = np.exp(la)
            beta = np.exp(lb)
            optimum_lr = np.exp(F) * n0**gamma * d0**zeta
            optimum_b = np.exp(G) * d0**eta
            return (
                C
                + A * d0 ** (-alpha)
                + B * n0 ** (-beta)
                + np.exp(Q) * (lr0 - optimum_lr) ** 2
                + np.exp(R) * (np.log(b0) + optimum_b / b0)
            )

        starts = []
        rng = np.random.default_rng(seed)
        for _ in range(18):
            starts.append(
                np.array(
                    [
                        rng.normal(1.0, 0.5),
                        rng.uniform(np.log(0.03), np.log(0.8)),
                        rng.normal(1.0, 0.5),
                        rng.uniform(np.log(0.03), np.log(0.8)),
                        np.quantile(y, 0.1),
                        rng.uniform(-5.0, 1.0),
                        rng.uniform(-5.0, 0.0),
                        rng.uniform(-2.0, 1.0),
                        rng.uniform(-1.5, 1.5),
                        rng.uniform(-1.5, 1.5),
                        rng.uniform(-2.0, 2.0),
                        rng.uniform(-1.0, 1.0),
                    ]
                )
            )
        arrays = (lrs, bs, ds, ns)
    else:
        # A compact log-coordinate response surface with explicit curvature.
        p_count = 12
        x_lr = np.log(lrs)
        x_b = np.log(bs)
        x_d = np.log(ds)
        x_n = np.log(ns)
        design = np.column_stack(
            [
                np.ones(len(y)),
                x_d,
                x_n,
                x_lr,
                x_b,
                x_lr**2,
                x_b**2,
                x_lr * x_d,
                x_lr * x_n,
                x_b * x_d,
                x_b * x_n,
                x_d * x_n,
            ]
        )
        beta = fit_linear_given_basis(design, y, ridge=1e-5)

        def predict_surface(df: pd.DataFrame) -> np.ndarray:
            lr1 = np.maximum(df["lr"].to_numpy(dtype=float), 1e-12) / scales["lr"]
            b1 = np.maximum(df["bsz"].to_numpy(dtype=float), 1.0) / scales["bsz"]
            d1 = (
                np.maximum(df["data_size"].to_numpy(dtype=float), 1.0)
                / scales["data"]
            )
            n1 = (
                np.maximum(
                    df["non_embedding_param_size"].to_numpy(dtype=float), 1.0
                )
                / scales["params"]
            )
            xl, xb, xd, xn = np.log(lr1), np.log(b1), np.log(d1), np.log(n1)
            xx = np.column_stack(
                [
                    np.ones(len(df)),
                    xd,
                    xn,
                    xl,
                    xb,
                    xl**2,
                    xb**2,
                    xl * xd,
                    xl * xn,
                    xb * xd,
                    xb * xn,
                    xd * xn,
                ]
            )
            return (xx @ beta)[:, None]

        return FittedLaw(
            name=variant,
            complexity=p_count,
            description="quadratic response surface in log-scaled hyperparameters",
            predict=predict_surface,
            parameters={"feature_scales": scales, "beta": beta.tolist()},
        )

    best = None
    for start in starts:
        result = least_squares(
            lambda theta: model(theta, arrays) - y,
            start,
            loss="soft_l1",
            f_scale=max(float(np.std(y)) * 0.1, 1e-4),
            max_nfev=6000,
        )
        mse = float(np.mean((model(result.x, arrays) - y) ** 2))
        if best is None or mse < best[0]:
            best = (mse, result.x)
    assert best is not None
    theta = best[1]

    def predict(df: pd.DataFrame) -> np.ndarray:
        arrays2 = (
            np.maximum(df["lr"].to_numpy(dtype=float), 1e-12) / scales["lr"],
            np.maximum(df["bsz"].to_numpy(dtype=float), 1.0) / scales["bsz"],
            np.maximum(df["data_size"].to_numpy(dtype=float), 1.0)
            / scales["data"],
            np.maximum(
                df["non_embedding_param_size"].to_numpy(dtype=float), 1.0
            )
            / scales["params"],
        )
        return model(theta, arrays2)[:, None]

    return FittedLaw(
        name=variant,
        complexity=p_count,
        description="hierarchical inverse-power loss with learned LR and batch optima",
        predict=predict,
        parameters={"feature_scales": scales, "theta": theta.tolist()},
    )


def candidate_factories(
    task_name: str,
    seed: int,
) -> list[tuple[str, Callable[[pd.DataFrame], FittedLaw]]]:
    if task_name == "sft_scaling_law":
        return [
            ("inverse_power", lambda df: fit_sft(df, "inverse_power")),
            (
                "saturating_inverse_power",
                lambda df: fit_sft(df, "saturating_inverse_power"),
            ),
        ]
    if task_name == "parallel_scaling_law":
        return [
            (
                "log_parallel_inverse_power",
                lambda df: fit_parallel(df, "log_parallel_inverse_power"),
            ),
            (
                "separable_inverse_power",
                lambda df: fit_parallel(df, "separable_inverse_power"),
            ),
        ]
    if task_name == "easy_question_scaling_law":
        return [
            (
                f"poly_{degree}_{alpha:g}",
                lambda df, degree=degree, alpha=alpha: fit_easy_question(
                    df, degree, alpha
                ),
            )
            for degree in range(1, 6)
            for alpha in (0.0, 1e-6, 1e-4, 1e-2)
        ]
    if task_name == "domain_mixture_scaling_law":
        polynomial = [
            (
                f"poly_{degree}_{alpha:g}",
                lambda df, degree=degree, alpha=alpha: fit_domain_mixture(
                    df, degree, alpha
                ),
            )
            for degree in (1, 2, 3)
            for alpha in (1e-8, 1e-5, 1e-3, 1e-2, 1e-1)
        ]
        return [
            ("mixture_exponential", fit_domain_mixture_exponential),
            *polynomial,
        ]
    if task_name in {
        "data_constrained_scaling_law",
        "vocab_scaling_law",
    }:
        return [
            (
                "additive_inverse_power",
                lambda df: fit_inverse_power_additive(
                    df, task_name, interaction=False, seed=seed
                ),
            ),
            (
                "additive_inverse_power_interaction",
                lambda df: fit_inverse_power_additive(
                    df, task_name, interaction=True, seed=seed
                ),
            ),
        ]
    if task_name == "moe_scaling_law":
        return [
            (
                "log_bilinear" if sat is None else f"log_bilinear_sat_{sat:g}",
                lambda df, sat=sat: fit_moe(df, sat),
            )
            for sat in (None, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
        ]
    if task_name == "lr_bsz_scaling_law":
        return [
            (
                "hierarchical_optimum",
                lambda df: fit_lr_bsz(df, "hierarchical_optimum", seed),
            ),
            (
                "log_quadratic_surface",
                lambda df: fit_lr_bsz(df, "log_quadratic_surface", seed),
            ),
        ]
    raise KeyError(task_name)


def run_task(
    task_name: str,
    train_full: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    val_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    fit, val = make_train_validation_split(train_full, task_name, val_fraction)
    candidates: list[CandidateResult] = []
    fitted_candidates: list[FittedLaw] = []
    for candidate_name, factory in candidate_factories(task_name, seed):
        try:
            law = factory(fit)
            val_nmse, stability_penalty, score = stable_selection_score(
                law, fit, val, task_name
            )
            candidates.append(
                CandidateResult(
                    name=law.name,
                    complexity=law.complexity,
                    val_nmse=val_nmse,
                    val_r2=1.0 - val_nmse,
                    stability_penalty=stability_penalty,
                    selection_score=score,
                )
            )
            fitted_candidates.append(law)
        except Exception as exc:
            candidates.append(
                CandidateResult(
                    name=candidate_name,
                    complexity=999,
                    val_nmse=100000.0,
                    val_r2=-99999.0,
                    stability_penalty=100000.0,
                    selection_score=200000.0,
                    fit_error=repr(exc),
                )
            )

    valid = [
        (record, law)
        for record, law in zip(
            [x for x in candidates if x.fit_error is None], fitted_candidates
        )
        if np.isfinite(record.selection_score)
    ]
    if not valid:
        raise RuntimeError(f"No valid candidates for {task_name}")
    selected_record, selected_fit_law = min(
        valid,
        key=lambda item: (
            item[0].selection_score,
            item[0].complexity,
            item[0].name,
        ),
    )

    # Freeze family/hyperparameters, then refit coefficients on all training rows.
    selected_factory = None
    for _, factory in candidate_factories(task_name, seed):
        probe = factory(fit)
        if probe.name == selected_fit_law.name:
            selected_factory = factory
            break
    if selected_factory is None:
        raise RuntimeError(f"Could not recover selected factory {selected_fit_law.name}")
    final_law = selected_factory(train_full)

    _, y_train = task_arrays(train_full, task_name)
    _, y_val = task_arrays(val, task_name)
    _, y_test = task_arrays(test, task_name)
    pred_train = final_law.predict(train_full)
    pred_val_frozen = final_law.predict(val)
    pred_test = final_law.predict(test)
    train_metrics = task_metrics(y_train, pred_train)
    frozen_val_metrics = task_metrics(y_val, pred_val_frozen)
    test_metrics = task_metrics(y_test, pred_test)

    result = {
        "seed": seed,
        "task": task_name,
        "method": "VEGA-SR-SLD-optimized",
        "selected_family": final_law.name,
        "family_description": final_law.description,
        "complexity_parameters": final_law.complexity,
        "n_train": len(train_full),
        "n_selection_fit": len(fit),
        "n_selection_val": len(val),
        "n_test": len(test),
        "selection_val_nmse": selected_record.val_nmse,
        "selection_val_r2": selected_record.val_r2,
        "selection_stability_penalty": selected_record.stability_penalty,
        "refit_train_nmse": train_metrics["nmse"],
        "refit_train_r2": train_metrics["r2"],
        "refit_frozen_val_nmse": frozen_val_metrics["nmse"],
        "refit_frozen_val_r2": frozen_val_metrics["r2"],
        "test_nmse": test_metrics["nmse"],
        "test_nmae": test_metrics["nmae"],
        "test_r2": test_metrics["r2"],
        "test_r2_clipped": test_metrics["clipped_r2"],
        "test_per_output_nmse": json.dumps(test_metrics["per_output_nmse"]),
        "parameters_json": json.dumps(final_law.parameters, sort_keys=True),
    }
    prediction_rows = test[["group", *TASK_SCHEMA[task_name]["features"]]].copy()
    prediction_rows.insert(0, "seed", seed)
    prediction_rows.insert(1, "task", task_name)
    for i, target in enumerate(TASK_SCHEMA[task_name]["targets"]):
        prediction_rows[f"true_{target}"] = y_test[:, i]
        prediction_rows[f"pred_{target}"] = pred_test[:, i]
        prediction_rows[f"residual_{target}"] = y_test[:, i] - pred_test[:, i]
    return result, [asdict(x) for x in candidates], prediction_rows


def load_split(data_root: Path, task_name: str, split: str) -> pd.DataFrame:
    path = data_root / task_name / f"{split}-00000-of-00001.parquet"
    frame = pd.read_parquet(path)
    frame["group"] = frame["group"].astype(str)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", default="11,23,47")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tasks", default=",".join(TASK_SCHEMA))
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    unknown = sorted(set(tasks) - set(TASK_SCHEMA))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / f"run_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    result_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for seed in seeds:
        for task_name in tasks:
            train = load_split(args.data_root, task_name, "train")
            test = load_split(args.data_root, task_name, "test")
            started = time.time()
            result, candidates, predictions = run_task(
                task_name,
                train_full=train,
                test=test,
                seed=seed,
                val_fraction=args.val_fraction,
            )
            result["runtime_sec"] = time.time() - started
            result_rows.append(result)
            for row in candidates:
                row.update({"seed": seed, "task": task_name})
                candidate_rows.append(row)
            prediction_frames.append(predictions)
            print(
                f"[{seed}] {task_name}: {result['selected_family']} "
                f"val_R2={result['selection_val_r2']:.4f} "
                f"test_R2={result['test_r2']:.4f} "
                f"clipped={result['test_r2_clipped']:.4f}"
            )

    results = pd.DataFrame(result_rows)
    candidates = pd.DataFrame(candidate_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    results.to_csv(run_root / "sldbench_optimized_task_results.csv", index=False)
    candidates.to_csv(run_root / "sldbench_candidate_selection.csv", index=False)
    predictions.to_csv(run_root / "sldbench_test_predictions.csv", index=False)

    summary = (
        results.groupby(["task", "method"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            median_test_r2=("test_r2", "median"),
            q25_test_r2=("test_r2", lambda x: x.quantile(0.25)),
            q75_test_r2=("test_r2", lambda x: x.quantile(0.75)),
            median_clipped_test_r2=("test_r2_clipped", "median"),
            median_test_nmse=("test_nmse", "median"),
            median_test_nmae=("test_nmae", "median"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .sort_values("task")
    )
    summary.to_csv(run_root / "sldbench_optimized_summary.csv", index=False)

    seed_summary = (
        results.groupby("seed", as_index=False)
        .agg(
            mean_task_r2=("test_r2", "mean"),
            median_task_r2=("test_r2", "median"),
            mean_clipped_task_r2=("test_r2_clipped", "mean"),
            tasks=("task", "nunique"),
        )
        .sort_values("seed")
    )
    seed_summary.to_csv(run_root / "sldbench_optimized_seed_summary.csv", index=False)

    comparison = summary.copy()
    comparison["human_reference_r2"] = comparison["task"].map(HUMAN_REFERENCE_R2)
    comparison["vega_minus_human"] = (
        comparison["median_clipped_test_r2"] - comparison["human_reference_r2"]
    )
    comparison["outcome_vs_human"] = np.where(
        comparison["vega_minus_human"] > 1e-6,
        "win",
        np.where(comparison["vega_minus_human"] < -1e-6, "loss", "tie"),
    )
    comparison.to_csv(
        run_root / "sldbench_optimized_vs_human_reference.csv", index=False
    )
    task_rng = np.random.default_rng(20260725)
    bootstrap_replicates = 100000
    task_indices = task_rng.integers(
        0,
        len(comparison),
        size=(bootstrap_replicates, len(comparison)),
    )
    vega_values = comparison["median_clipped_test_r2"].to_numpy(dtype=float)
    human_values = comparison["human_reference_r2"].to_numpy(dtype=float)
    vega_bootstrap = vega_values[task_indices].mean(axis=1)
    paired_difference = (
        vega_values[task_indices] - human_values[task_indices]
    ).mean(axis=1)
    comparison_summary = {
        "vega_mean_clipped_task_r2": float(np.mean(vega_values)),
        "human_reference_mean_task_r2": float(np.mean(human_values)),
        "absolute_difference": float(np.mean(vega_values - human_values)),
        "relative_improvement_percent": float(
            100.0
            * np.mean(vega_values - human_values)
            / max(abs(float(np.mean(human_values))), 1e-12)
        ),
        "wins": int((comparison["outcome_vs_human"] == "win").sum()),
        "ties": int((comparison["outcome_vs_human"] == "tie").sum()),
        "losses": int((comparison["outcome_vs_human"] == "loss").sum()),
        "bootstrap_unit": "official task",
        "bootstrap_replicates": bootstrap_replicates,
        "vega_mean_clipped_task_r2_95ci": [
            float(x) for x in np.quantile(vega_bootstrap, [0.025, 0.975])
        ],
        "paired_mean_difference_95ci": [
            float(x) for x in np.quantile(paired_difference, [0.025, 0.975])
        ],
        "bootstrap_probability_difference_gt_zero": float(
            np.mean(paired_difference > 0.0)
        ),
        "note": (
            "Human reference values are transcribed from the public Harbor "
            "SLDBench adapter README; task R2 is clipped to [-1, 1]."
        ),
    }
    (run_root / "sldbench_comparison_summary.json").write_text(
        json.dumps(comparison_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    audit = {
        "status": "exploratory_public_test",
        "method": "VEGA-SR-SLD-optimized",
        "task_unit": "8 official SLDBench tasks",
        "selection_data": "official train split only",
        "selection_split": "group-wise largest-scale holdout",
        "final_evaluation_data": "public official test split",
        "metrics": "task-level R2 = 1 - mean per-output NMSE; NMAE",
        "candidate_priors": (
            "documented generic scaling-law families; all numerical coefficients "
            "fit from train"
        ),
        "seeds": seeds,
        "val_fraction": args.val_fraction,
        "tasks": tasks,
        "data_root": str(args.data_root.resolve()),
    }
    (run_root / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_root / "LATEST").write_text(str(run_root.resolve()), encoding="utf-8")

    print("\nTask summary")
    print(summary.to_string(index=False))
    print("\nSeed summary")
    print(seed_summary.to_string(index=False))
    print(f"\nSaved: {run_root}")


if __name__ == "__main__":
    main()
