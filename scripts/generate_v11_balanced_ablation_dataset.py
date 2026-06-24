#!/usr/bin/env python3
"""Generate a moderate unified V11 component-ablation dataset.

This dataset is intentionally less adversarial than the hard observer-only
proxy-shift suite. All methods run on the same cases. The cases use randomized
active-variable positions, weak correlated proxies, nonlinear decoys, and
independent distractors without changing the test distribution.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from run_v11_high_dimensional_interference import FORMULA_BY_ID, render_expression
except ModuleNotFoundError:
    from scripts.run_v11_high_dimensional_interference import FORMULA_BY_ID, render_expression


FORMULA_IDS = [
    "feynman_trig_interaction_k2",
    "nguyen_rational_k2",
    "livermore_exp_trig_k2",
    "srsd_log_sqrt_k2",
    "feynman_sparse_interaction_k3",
    "friedman_core_k3",
    "srsd_exp_log_k3",
    "nguyen_trig_poly_k3",
    "feynman_rational_trig_k4",
    "friedman2_safe_k4",
    "srsd_energy_like_k4",
    "livermore_rational_interaction_k4",
]


@dataclass(frozen=True)
class CaseSpec:
    case_name: str
    formula_id: str
    dimension: int
    seed: int
    proxy_strength: float
    decoy_noise: float


def sanitize(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text)).strip("_") or "case"


def dimension_for(true_count: int, variant: int) -> int:
    base = 12 if true_count <= 2 else 16 if true_count == 3 else 20
    return base + 4 * (variant % 3)


def decoy_count_for(true_count: int, dimension: int) -> int:
    return min(max(true_count + 2, 4), 8, dimension - true_count)


def make_cases(num_cases: int) -> list[CaseSpec]:
    if num_cases <= 0:
        raise ValueError("num_cases must be positive")
    repeats = (num_cases + len(FORMULA_IDS) - 1) // len(FORMULA_IDS)
    formula_cycle = (FORMULA_IDS * repeats)[:num_cases]
    cases: list[CaseSpec] = []
    for idx, formula_id in enumerate(formula_cycle, start=1):
        formula = FORMULA_BY_ID[formula_id]
        variant = (idx - 1) // len(FORMULA_IDS)
        cases.append(
            CaseSpec(
                case_name=f"balanced_{idx:03d}_{formula_id}",
                formula_id=formula_id,
                dimension=dimension_for(int(formula.true_variable_count), variant),
                seed=50_000 + idx,
                proxy_strength=[0.25, 0.35, 0.45, 0.55][variant % 4],
                decoy_noise=[0.10, 0.14, 0.18][variant % 3],
            )
        )
    return cases


def feature_layout(spec: CaseSpec) -> tuple[list[str], list[str], list[str], list[str]]:
    formula = FORMULA_BY_ID[spec.formula_id]
    true_count = int(formula.true_variable_count)
    feature_names = [f"x{i}" for i in range(1, spec.dimension + 1)]
    rng = np.random.default_rng(spec.seed + 17)

    # Keep active variables randomized but not buried impossibly deep.
    pool = np.arange(spec.dimension)
    true_positions = sorted(rng.choice(pool, size=true_count, replace=False).tolist())
    true_variables = [feature_names[idx] for idx in true_positions]

    remaining = [idx for idx in range(spec.dimension) if idx not in set(true_positions)]
    decoy_count = decoy_count_for(true_count, spec.dimension)
    proxy_count = max(1, min(true_count, decoy_count // 2))
    proxy_positions = remaining[:proxy_count]
    nonlinear_positions = remaining[proxy_count:decoy_count]
    proxy_variables = [feature_names[idx] for idx in proxy_positions]
    nonlinear_decoy_variables = [feature_names[idx] for idx in nonlinear_positions]
    return feature_names, true_variables, proxy_variables, nonlinear_decoy_variables


def make_split(spec: CaseSpec, n: int, rng: np.random.Generator) -> pd.DataFrame:
    formula = FORMULA_BY_ID[spec.formula_id]
    true_count = int(formula.true_variable_count)
    low = np.asarray(formula.input_low, dtype=float)
    high = np.asarray(formula.input_high, dtype=float)
    feature_names, true_variables, proxy_variables, nonlinear_decoy_variables = feature_layout(spec)
    name_to_col = {name: idx for idx, name in enumerate(feature_names)}

    z = rng.uniform(low, high, size=(n, true_count))
    x = rng.normal(0.0, 1.0, size=(n, spec.dimension))
    for local_idx, name in enumerate(true_variables):
        x[:, name_to_col[name]] = z[:, local_idx]

    for local_idx, name in enumerate(proxy_variables):
        source = z[:, local_idx % true_count]
        noise = rng.normal(0.0, max(0.05, np.std(source)), size=n)
        x[:, name_to_col[name]] = spec.proxy_strength * source + (1.0 - spec.proxy_strength) * noise

    transforms = [
        lambda v: np.sin(v),
        lambda v: np.cos(v),
        lambda v: v**2,
        lambda v: np.tanh(v),
        lambda v: np.sqrt(np.abs(v) + 1.0e-6),
    ]
    for local_idx, name in enumerate(nonlinear_decoy_variables):
        source = z[:, local_idx % true_count]
        x[:, name_to_col[name]] = transforms[local_idx % len(transforms)](source) + spec.decoy_noise * rng.normal(size=n)

    y = formula.fn(z)
    df = pd.DataFrame(x, columns=feature_names)
    df["y"] = np.asarray(y, dtype=float)
    return df


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def write_case(root: Path, rows: list[dict], idx: int, spec: CaseSpec, n_train: int, n_val: int, n_test: int) -> None:
    formula = FORMULA_BY_ID[spec.formula_id]
    feature_names, true_variables, proxy_variables, nonlinear_decoy_variables = feature_layout(spec)
    case_id = sanitize(spec.case_name)
    case_dir = root / "suites" / "component_ablation" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for split, n, offset in [("train", n_train, 1), ("val", n_val, 2), ("test", n_test, 3)]:
        df = make_split(spec, n, np.random.default_rng(spec.seed * 100 + offset))
        df.to_csv(case_dir / f"{split}.csv", index=False)

    row = {
        "suite": "component_ablation",
        "suite_case_index": idx,
        "case_id": case_id,
        "case_name": spec.case_name,
        "benchmark": formula.source_family,
        "source_name": formula.source_name,
        "base_formula_id": formula.formula_id,
        "target_component": "balanced",
        "failure_mode": "moderate_unified_component_ablation",
        "structure_type": formula.structure_type,
        "feature_names": "|".join(feature_names),
        "true_variables": "|".join(true_variables),
        "true_variable_positions": "|".join(str(feature_names.index(name) + 1) for name in true_variables),
        "true_expression": render_expression(formula, true_variables),
        "fn_name": formula.formula_id,
        "noise_level": 0.0,
        "dimension": spec.dimension,
        "true_variable_count": formula.true_variable_count,
        "interference_type": "moderate_mixed_distractors",
        "proxy_variables": "|".join(proxy_variables),
        "proxy_variable_positions": "|".join(str(feature_names.index(name) + 1) for name in proxy_variables),
        "nonlinear_decoy_variables": "|".join(nonlinear_decoy_variables),
        "nonlinear_decoy_positions": "|".join(str(feature_names.index(name) + 1) for name in nonlinear_decoy_variables),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": spec.seed,
        "proxy_strength": spec.proxy_strength,
        "decoy_noise": spec.decoy_noise,
        "split_dir": rel(case_dir, root),
        "train_path": rel(case_dir / "train.csv", root),
        "val_path": rel(case_dir / "val.csv", root),
        "test_path": rel(case_dir / "test.csv", root),
        "train_clean_path": "",
        "val_clean_path": "",
    }
    (case_dir / "meta.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    rows.append(row)


def generate(args: argparse.Namespace) -> None:
    root = Path(args.out_dir).resolve()
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cases = make_cases(args.num_cases)
    for idx, spec in enumerate(cases, start=1):
        write_case(root, rows, idx, spec, args.n_train, args.n_val, args.n_test)
    manifest = pd.DataFrame(rows)
    manifest.insert(0, "global_case_index", np.arange(1, len(manifest) + 1))
    manifest.to_csv(root / "manifest.csv", index=False)
    (root / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(x) for x in cases]).to_csv(root / "balanced_specs.csv", index=False)
    readme = {
        "name": f"v11_balanced_component_ablation_{len(rows)}",
        "case_count": len(rows),
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "note": "Moderate unified component-ablation set: randomized true-variable positions, weak proxies, nonlinear decoys, no proxy test shift.",
        "structure_counts": manifest.groupby("structure_type").size().to_dict(),
        "true_variable_counts": manifest.groupby("true_variable_count").size().to_dict(),
    }
    (root / "README.md").write_text(json.dumps(readme, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[dataset] wrote {len(rows)} cases to {root}")
    print(manifest.groupby(["true_variable_count", "dimension"]).size().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/mnt/hdd/llmsr_data/v11_balanced_component_ablation_96")
    parser.add_argument("--num-cases", type=int, default=96)
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-val", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    generate(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
