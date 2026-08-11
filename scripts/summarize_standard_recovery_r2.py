#!/usr/bin/env python3
"""Build canonical test-R2 results for the paper's standard-recovery methods."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tools.plot_style import METHOD_COLORS, NATURE_COLORS, save_nature_figure, set_nature_style


EXPECTED = {
    "llmsrbench": 240,
    "sldbench": 77,
    "srbench": 417,
    "srsd": 238,
}
METHOD_MAP = {
    "dso": "DSO",
    "gplearn": "gplearn",
    "psrn_pse": "PSE",
    "pysr": "PySR",
    "official_icsr_adapted100": "ICSR",
}
METHOD_ORDER = ["DSO", "gplearn", "PSE", "PySR", "ICSR"]
BENCHMARK_ORDER = ["llmsrbench", "sldbench", "srbench", "srsd"]
METHOD_COLOR = {
    "DSO": METHOD_COLORS["dso"],
    "gplearn": METHOD_COLORS["gplearn"],
    "PSE": METHOD_COLORS["pse"],
    "PySR": METHOD_COLORS["pysr"],
    "ICSR": METHOD_COLORS["icsr"],
}


def import_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalize_srsd_case(value: str) -> str:
    text = str(value).replace("\\", "/")
    match = re.search(r"/srsd/([^/]+)/true_eq/([^/]+)\.pkl$", text)
    if match:
        return f"{match.group(1)}::{match.group(2)}"
    return text


def canonical_case_id(
    benchmark: str,
    case_name: str,
    dataset_group: str | None = None,
) -> str:
    text = str(case_name).strip().replace("\\", "/")
    if benchmark == "srsd":
        normalized = normalize_srsd_case(text)
        if normalized != text:
            return normalized
        parts = [part.strip() for part in text.split(" | ")]
        if len(parts) >= 3 and parts[0].lower() == "srsd":
            return f"{parts[-2]}::{parts[-1]}"
        group = str(dataset_group).strip()
        if group and group.lower() != "nan":
            case_id = text[:-4] if text.endswith(".pkl") else text
            return f"{group}::{case_id}"
        return text[:-4] if text.endswith(".pkl") else text
    return text.split(" | ")[-1].strip()


def case_key(
    benchmark: str,
    case_name: str,
    dataset_group: str | None = None,
) -> str:
    return f"{benchmark}::{canonical_case_id(benchmark, case_name, dataset_group)}"


def target_variance(test_df: pd.DataFrame) -> tuple[float, int]:
    y = pd.to_numeric(test_df.iloc[:, -1], errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return np.nan, 0
    return float(np.mean(np.square(y - np.mean(y)))), int(len(y))


def checkpoint_variances(rows: list[dict], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates("case_key", keep="last").to_csv(cache_path, index=False)


def build_variance_cache(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        existing = pd.read_csv(cache_path)
        existing["case_key"] = [
            case_key(benchmark, case_name)
            for benchmark, case_name in zip(existing["benchmark"], existing["case_name"])
        ]
        existing = existing.drop_duplicates("case_key", keep="last")
        rows = existing.to_dict("records")
        completed = set(existing["case_key"].astype(str))
    else:
        rows = []
        completed = set()

    os.environ.setdefault(
        "LLMSRBENCH_CASES_ROOT",
        str(ROOT / "data" / "sim-datasets-llmsr" / "llm-srbench"),
    )
    os.environ.setdefault("SLDBENCH_HUB_REPO", str(ROOT / "data" / "sldbench_repo"))
    os.environ.setdefault("SRBENCH_ROOT", str(ROOT / "srbench"))
    os.environ.setdefault(
        "SRBENCH_DATASETS_INFO_CSV",
        str(ROOT / "srbench" / "docs" / "csv" / "datasets_info.csv"),
    )
    os.environ.setdefault("SRBENCH_PMLB_CACHE_DIR", str(ROOT / ".cache" / "pmlb_cache"))
    os.environ.setdefault(
        "SRSD_ROOT",
        str(ROOT / "srsd-benchmark" / "resource" / "datasets" / "srsd"),
    )

    llmsr = import_script("r2_llmsrbench_loader", ROOT / "scripts" / "run_llmsrbench.py")
    for _, row in llmsr.collect_llmsrbench_tasks().iterrows():
        row_dict = row.to_dict()
        train_df, val_df, test_df, extra = llmsr.load_llmsrbench_case(row_dict)
        name = (
            f"llmsrbench | {row_dict['split_name']} | "
            f"{extra['target_name_original']} | {row_dict['case_name']}"
        )
        key = case_key("llmsrbench", name)
        if key in completed:
            continue
        variance, n_test = target_variance(test_df)
        rows.append({
            "benchmark": "llmsrbench",
            "case_name": name,
            "case_key": key,
            "target_variance": variance,
            "n_test": n_test,
            "variance_source": extra.get("test_source", extra.get("layout_source", "official_test")),
        })
        completed.add(key)
    checkpoint_variances(rows, cache_path)

    sld = import_script("r2_sldbench_loader", ROOT / "scripts" / "run_sldbench.py")
    for idx, (_, row) in enumerate(sld.collect_sldbench_tasks().iterrows(), start=1):
        row_dict = row.to_dict()
        _, _, test_df = sld.load_sldbench_case(row_dict)
        name = f"sldbench | {row_dict['task_name']} | {row_dict['case_name']}"
        key = case_key("sldbench", name)
        if key in completed:
            continue
        variance, n_test = target_variance(test_df)
        rows.append({
            "benchmark": "sldbench",
            "case_name": name,
            "case_key": key,
            "target_variance": variance,
            "n_test": n_test,
            "variance_source": "official_test",
        })
        completed.add(key)
        if idx % 10 == 0:
            checkpoint_variances(rows, cache_path)
    checkpoint_variances(rows, cache_path)

    srsd = import_script("r2_srsd_loader", ROOT / "scripts" / "run_srds.py")
    frames = []
    for dataset_dir in srsd.SRSD_DATASET_DIRS:
        frame = srsd.collect_raw_tasks_for_dataset(dataset_dir)
        if len(frame):
            frames.append(frame)
    srsd_tasks = pd.concat(frames, ignore_index=True)
    for idx, (_, row) in enumerate(srsd_tasks.iterrows(), start=1):
        row_dict = row.to_dict()
        name = str(row_dict["true_eq_path"])
        key = case_key("srsd", name)
        if key in completed:
            continue
        test_df = srsd.load_txt_dataset(row_dict["test_path"])
        variance, n_test = target_variance(test_df)
        rows.append({
            "benchmark": "srsd",
            "case_name": name,
            "case_key": key,
            "target_variance": variance,
            "n_test": n_test,
            "variance_source": "official_test",
        })
        completed.add(key)
        if idx % 25 == 0:
            checkpoint_variances(rows, cache_path)
    checkpoint_variances(rows, cache_path)

    srbench = import_script("r2_srbench_loader", ROOT / "scripts" / "run_SRbench.py")
    auxiliary_cache = Path(os.environ.get(
        "SRBENCH_AUX_PMLB_CACHE_DIR",
        str(ROOT.parent / "PSE" / "pmlb_cache"),
    ))
    proxy_names = (
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"
    )
    old_proxy = {name: os.environ.pop(name, None) for name in proxy_names}
    try:
        tasks = srbench.collect_srbench_tasks()
        for idx, (_, row) in enumerate(tasks.iterrows(), start=1):
            row_dict = row.to_dict()
            name = f"srbench | {row_dict['group_name']} | {row_dict['dataset_name']}"
            key = case_key("srbench", name)
            if key in completed:
                continue
            auxiliary_path = (
                auxiliary_cache / row_dict["dataset_name"] / f"{row_dict['dataset_name']}.tsv.gz"
            )
            if auxiliary_path.exists():
                raw_df = pd.read_csv(auxiliary_path, sep="\t", compression="gzip")
                source_kind = "auxiliary_pmlb_cache_tsv"
            else:
                raw_df, source_kind, _ = srbench.load_raw_dataset_frame(row_dict["dataset_name"])
            xy_df, _ = srbench.make_numeric_xy_dataframe(raw_df)
            _, _, test_df = srbench.split_train_val_test(
                xy_df,
                test_ratio=srbench.TEST_RATIO,
                val_ratio_within_trainval=srbench.VAL_RATIO_WITHIN_TRAINVAL,
                seed=srbench.TRAIN_TEST_SPLIT_SEED,
                shuffle=srbench.SPLIT_SHUFFLE,
            )
            variance, n_test = target_variance(test_df)
            rows.append({
                "benchmark": "srbench",
                "case_name": name,
                "case_key": key,
                "target_variance": variance,
                "n_test": n_test,
                "variance_source": source_kind,
            })
            completed.add(key)
            if idx % 10 == 0:
                checkpoint_variances(rows, cache_path)
    finally:
        for name, value in old_proxy.items():
            if value is not None:
                os.environ[name] = value
    checkpoint_variances(rows, cache_path)
    return pd.read_csv(cache_path)


def load_paper_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["method"].isin(METHOD_MAP)].copy()
    frame["method_id"] = frame["method"]
    frame["method"] = frame["method"].map(METHOD_MAP)
    frame["benchmark"] = frame["benchmark"].astype(str).str.lower()
    frame["case_key"] = [
        case_key(benchmark, case_name, dataset_group)
        for benchmark, case_name, dataset_group in zip(
            frame["benchmark"], frame["case_name"], frame["dataset_group"]
        )
    ]
    frame["test_mse"] = pd.to_numeric(frame["test_mse"], errors="coerce")
    return frame


def derive_r2(case_rows: pd.DataFrame, variances: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_key", "target_variance", "n_test", "variance_source"
    ]
    out = case_rows.merge(variances[columns], how="left", on="case_key", validate="many_to_one")
    valid = (
        np.isfinite(out["test_mse"])
        & np.isfinite(out["target_variance"])
        & out["target_variance"].gt(0)
    )
    out["test_r2"] = np.nan
    out.loc[valid, "test_r2"] = (
        1.0 - out.loc[valid, "test_mse"] / out.loc[valid, "target_variance"]
    )
    out["r2_available"] = np.isfinite(out["test_r2"])
    out["r2_status"] = "available"
    out.loc[~np.isfinite(out["test_mse"]), "r2_status"] = "missing_test_mse"
    out.loc[
        np.isfinite(out["test_mse"]) & ~np.isfinite(out["target_variance"]),
        "r2_status",
    ] = "missing_target_variance"
    out.loc[
        np.isfinite(out["test_mse"])
        & np.isfinite(out["target_variance"])
        & out["target_variance"].le(0),
        "r2_status",
    ] = "nonpositive_target_variance"
    out["numerical_complete_fit_r2"] = out["test_r2"].gt(0.999)
    out["r2_source"] = "canonical_test_mse_and_target_variance"
    return out


def summarize_group(frame: pd.DataFrame, method: str, benchmark: str) -> dict:
    finite = pd.to_numeric(frame["test_r2"], errors="coerce")
    finite = finite[np.isfinite(finite)]
    test_mse = pd.to_numeric(frame["test_mse"], errors="coerce")
    finite_mse = np.isfinite(test_mse)
    target_variance = pd.to_numeric(frame["target_variance"], errors="coerce")
    expected_n = EXPECTED[benchmark] if benchmark != "overall" else sum(EXPECTED.values())
    return {
        "method": method,
        "benchmark": benchmark,
        "expected_n": expected_n,
        "paper_row_n": int(len(frame)),
        "missing_paper_row_n": int(expected_n - len(frame)),
        "finite_test_mse_n": int(finite_mse.sum()),
        "missing_test_mse_n": int((~finite_mse).sum()),
        "nonpositive_target_variance_n": int(
            (finite_mse & np.isfinite(target_variance) & target_variance.le(0)).sum()
        ),
        "finite_r2_n": int(len(finite)),
        "finite_r2_coverage_expected": float(len(finite) / expected_n),
        "missing_r2_n": int(expected_n - len(finite)),
        "mean_test_r2_finite": float(finite.mean()) if len(finite) else np.nan,
        "median_test_r2_finite": float(finite.median()) if len(finite) else np.nan,
        "q25_test_r2_finite": float(finite.quantile(0.25)) if len(finite) else np.nan,
        "q75_test_r2_finite": float(finite.quantile(0.75)) if len(finite) else np.nan,
        "r2_gt_0_999_n": int((finite > 0.999).sum()),
        "r2_gt_0_999_rate_expected": float((finite > 0.999).sum() / expected_n),
        "r2_gt_0_n": int((finite > 0).sum()),
        "r2_gt_0_rate_expected": float((finite > 0).sum() / expected_n),
    }


def summarize(case_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        method_rows = case_rows[case_rows["method"] == method]
        for benchmark in BENCHMARK_ORDER:
            rows.append(summarize_group(
                method_rows[method_rows["benchmark"] == benchmark],
                method,
                benchmark,
            ))
        rows.append(summarize_group(method_rows, method, "overall"))
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    set_nature_style(plt=plt)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.25, 1.0]})

    heat = np.full((len(BENCHMARK_ORDER), len(METHOD_ORDER)), np.nan)
    coverage = np.zeros_like(heat)
    for i, benchmark in enumerate(BENCHMARK_ORDER):
        for j, method in enumerate(METHOD_ORDER):
            row = summary[
                (summary["benchmark"] == benchmark) & (summary["method"] == method)
            ].iloc[0]
            heat[i, j] = row["median_test_r2_finite"]
            coverage[i, j] = row["finite_r2_coverage_expected"]
    display_heat = np.clip(heat, -1.0, 1.0)
    im = axes[0].imshow(display_heat, cmap="RdYlBu", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(METHOD_ORDER)), METHOD_ORDER, rotation=25, ha="right")
    axes[0].set_yticks(
        range(len(BENCHMARK_ORDER)),
        ["LLMSRBench", "SLDBench", "SRBench", "SRSD"],
    )
    axes[0].set_title("Median test $R^2$ (finite results)")
    for i in range(len(BENCHMARK_ORDER)):
        for j in range(len(METHOD_ORDER)):
            value = heat[i, j]
            label = "NA" if not np.isfinite(value) else f"{value:.2f}\n{coverage[i, j]:.0%}"
            axes[0].text(
                j, i, label, ha="center", va="center", fontsize=6.2,
                color="white" if np.isfinite(display_heat[i, j]) and abs(display_heat[i, j]) > 0.55 else NATURE_COLORS["dark"],
            )
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.035)
    cbar.set_label("Median test $R^2$ (color clipped to [-1, 1])")

    overall = summary[summary["benchmark"] == "overall"].set_index("method").loc[METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER))
    fit_rate = 100.0 * overall["r2_gt_0_999_rate_expected"].to_numpy(dtype=float)
    cov_rate = 100.0 * overall["finite_r2_coverage_expected"].to_numpy(dtype=float)
    axes[1].bar(
        x, cov_rate, width=0.7, color=NATURE_COLORS["neutral"],
        edgecolor=NATURE_COLORS["midgray"], linewidth=0.6, label="Finite $R^2$ coverage",
    )
    axes[1].bar(
        x, fit_rate, width=0.7,
        color=[METHOD_COLOR[m] for m in METHOD_ORDER],
        edgecolor="white", linewidth=0.5, label="$R^2 > 0.999$",
    )
    axes[1].set_xticks(x, METHOD_ORDER, rotation=25, ha="right")
    axes[1].set_ylabel("Share of 972 tasks (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Numerical complete fit and coverage")
    axes[1].grid(axis="y", alpha=0.35)
    axes[1].legend(loc="upper right")

    for ax, panel in zip(axes, ("a", "b")):
        ax.text(-0.13, 1.05, panel, transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.tight_layout()
    png = output_dir / "fig_standard_recovery_other_methods_test_r2.png"
    pdf = output_dir / "fig_standard_recovery_other_methods_test_r2.pdf"
    save_nature_figure(fig, png, section="standard_recovery", dpi=300, bbox_inches="tight")
    save_nature_figure(fig, pdf, section="standard_recovery", bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-case-rows",
        type=Path,
        default=ROOT / "dxevolve-nature-latex" / "supplement_submission_assets_3"
        / "source_data" / "fig1" / "fig1_case_rows.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variance-cache", type=Path, default=None)
    parser.add_argument("--build-variance-cache", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (args.variance_cache or output_dir / "standard_recovery_test_target_variances.csv").resolve()
    if args.build_variance_cache:
        variances = build_variance_cache(cache_path)
    elif cache_path.exists():
        variances = pd.read_csv(cache_path)
    else:
        raise FileNotFoundError(
            f"Variance cache not found: {cache_path}. Run once with --build-variance-cache."
        )

    paper_rows = load_paper_rows(args.paper_case_rows)
    case_rows = derive_r2(paper_rows, variances)
    summary = summarize(case_rows)
    case_rows.to_csv(output_dir / "standard_recovery_other_methods_r2_case_rows.csv", index=False)
    summary.to_csv(output_dir / "standard_recovery_other_methods_r2_summary.csv", index=False)
    plot_summary(summary, output_dir)
    print(summary.to_string(index=False))
    print(f"[INFO] variance cases: {len(variances)} / {sum(EXPECTED.values())}")
    print(f"[INFO] R2 case rows: {int(case_rows['r2_available'].sum())} / {len(case_rows)}")
    print(f"[INFO] outputs: {output_dir}")


if __name__ == "__main__":
    main()
