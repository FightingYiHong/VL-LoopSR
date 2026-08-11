#!/usr/bin/env python3
"""Generate the VEGA-SR claim-validation component-ablation dataset.

The original balanced component-ablation generator creates one generic
``balanced`` target.  Claim validation needs two paired target components:
``observer`` for visual-causal tests and ``critic`` for feedback-causal tests.
This generator keeps the same formula and distractor machinery but writes a
manifest with target-specific case groups and the requested sample counts.
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
    from generate_v11_balanced_ablation_dataset import FORMULA_IDS
    from generate_v11_balanced_ablation_dataset import CaseSpec as BaseCaseSpec
    from generate_v11_balanced_ablation_dataset import dimension_for, feature_layout, make_split, rel, sanitize
    from run_v11_high_dimensional_interference import FORMULA_BY_ID, render_expression
except ModuleNotFoundError:
    from scripts.generate_v11_balanced_ablation_dataset import FORMULA_IDS
    from scripts.generate_v11_balanced_ablation_dataset import CaseSpec as BaseCaseSpec
    from scripts.generate_v11_balanced_ablation_dataset import dimension_for, feature_layout, make_split, rel, sanitize
    from scripts.run_v11_high_dimensional_interference import FORMULA_BY_ID, render_expression


TARGET_SEED_OFFSETS = {
    "observer": 70_000,
    "critic": 90_000,
}


@dataclass(frozen=True)
class ClaimCaseSpec:
    target_component: str
    target_case_index: int
    base: BaseCaseSpec


def split_targets(text: str) -> list[str]:
    out = [x.strip() for x in re.split(r"[, ]+", str(text or "")) if x.strip()]
    return out or ["observer", "critic"]


def make_target_cases(target: str, cases_per_target: int) -> list[ClaimCaseSpec]:
    if cases_per_target <= 0:
        raise ValueError("cases_per_target must be positive")
    repeats = (cases_per_target + len(FORMULA_IDS) - 1) // len(FORMULA_IDS)
    formula_cycle = (FORMULA_IDS * repeats)[:cases_per_target]
    seed_offset = TARGET_SEED_OFFSETS.get(target, 110_000)
    cases: list[ClaimCaseSpec] = []
    for idx, formula_id in enumerate(formula_cycle, start=1):
        formula = FORMULA_BY_ID[formula_id]
        variant = (idx - 1) // len(FORMULA_IDS)
        case_name = f"claim_{target}_{idx:03d}_{formula_id}"
        cases.append(
            ClaimCaseSpec(
                target_component=target,
                target_case_index=idx,
                base=BaseCaseSpec(
                    case_name=case_name,
                    formula_id=formula_id,
                    dimension=dimension_for(int(formula.true_variable_count), variant),
                    seed=seed_offset + idx,
                    proxy_strength=[0.25, 0.35, 0.45, 0.55][variant % 4],
                    decoy_noise=[0.10, 0.14, 0.18][variant % 3],
                ),
            )
        )
    return cases


def write_case(
    root: Path,
    rows: list[dict],
    global_idx: int,
    suite_idx: int,
    spec: ClaimCaseSpec,
    n_train: int,
    n_val: int,
    n_test: int,
) -> None:
    base = spec.base
    formula = FORMULA_BY_ID[base.formula_id]
    feature_names, true_variables, proxy_variables, nonlinear_decoy_variables = feature_layout(base)
    case_id = sanitize(base.case_name)
    case_dir = root / "suites" / "component_ablation" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    for split, n, offset in [("train", n_train, 1), ("val", n_val, 2), ("test", n_test, 3)]:
        df = make_split(base, n, np.random.default_rng(base.seed * 100 + offset))
        df.to_csv(case_dir / f"{split}.csv", index=False)

    row = {
        "global_case_index": global_idx,
        "suite": "component_ablation",
        "suite_case_index": suite_idx,
        "case_id": case_id,
        "case_name": base.case_name,
        "benchmark": formula.source_family,
        "source_name": formula.source_name,
        "base_formula_id": formula.formula_id,
        "target_component": spec.target_component,
        "target_case_index": spec.target_case_index,
        "failure_mode": f"claim_validation_{spec.target_component}",
        "structure_type": formula.structure_type,
        "feature_names": "|".join(feature_names),
        "true_variables": "|".join(true_variables),
        "true_variable_positions": "|".join(str(feature_names.index(name) + 1) for name in true_variables),
        "true_expression": render_expression(formula, true_variables),
        "fn_name": formula.formula_id,
        "noise_level": 0.0,
        "dimension": base.dimension,
        "true_variable_count": formula.true_variable_count,
        "interference_type": "claim_validation_mixed_distractors",
        "proxy_variables": "|".join(proxy_variables),
        "proxy_variable_positions": "|".join(str(feature_names.index(name) + 1) for name in proxy_variables),
        "nonlinear_decoy_variables": "|".join(nonlinear_decoy_variables),
        "nonlinear_decoy_positions": "|".join(str(feature_names.index(name) + 1) for name in nonlinear_decoy_variables),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": base.seed,
        "proxy_strength": base.proxy_strength,
        "decoy_noise": base.decoy_noise,
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

    targets = split_targets(args.targets)
    rows: list[dict] = []
    global_idx = 1
    suite_idx = 1
    all_specs: list[dict] = []
    for target in targets:
        for spec in make_target_cases(target, args.cases_per_target):
            write_case(root, rows, global_idx, suite_idx, spec, args.n_train, args.n_val, args.n_test)
            all_specs.append({
                "target_component": target,
                "target_case_index": spec.target_case_index,
                **asdict(spec.base),
            })
            global_idx += 1
            suite_idx += 1

    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / "manifest.csv", index=False)
    (root / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(all_specs).to_csv(root / "claim_validation_specs.csv", index=False)

    structure_counts = {
        "|".join(str(x) for x in key): int(value)
        for key, value in manifest.groupby(["target_component", "structure_type"]).size().to_dict().items()
    }
    true_variable_counts = {
        "|".join(str(x) for x in key): int(value)
        for key, value in manifest.groupby(["target_component", "true_variable_count"]).size().to_dict().items()
    }
    readme = {
        "name": "vega_sr_claim_validation_v1",
        "case_count": len(rows),
        "targets": targets,
        "cases_per_target": args.cases_per_target,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "note": "VEGA-SR causal claim-validation dataset with observer and critic target groups.",
        "target_counts": manifest.groupby("target_component").size().to_dict(),
        "structure_counts": structure_counts,
        "true_variable_counts": true_variable_counts,
    }
    (root / "README.md").write_text(json.dumps(readme, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[dataset] wrote {len(rows)} cases to {root}")
    print(manifest.groupby(["target_component", "true_variable_count", "dimension"]).size().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cases-per-target", type=int, default=24)
    parser.add_argument("--targets", default="observer,critic")
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-val", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    generate(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
