#!/usr/bin/env python3
"""Download or generate the datasets required by the paper experiments.

Large public datasets and third-party repositories intentionally live under
``data/external`` or ``data/generated`` and are not committed to this repo.
The bundled Proposer SFT corpus is validated in place. Existing targets are
never overwritten.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
GENERATED = ROOT / "data" / "generated"

SRBENCH_REV = "5b456d5108522c4da89aded2f07e0fb2bee4a3a4"
SRSD_REV = "7d00b45d56250717ac08a256dc9f79b836d61027"
LLMSRBENCH_REV = "4f22d48100d94125d3af1e7bd40897fcc7f9597d"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_at(url: str, target: Path, revision: str) -> None:
    if (target / ".git").is_dir():
        current = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=target, text=True
        ).strip()
        if current != revision:
            raise SystemExit(
                f"Existing checkout {target} is at {current}; expected {revision}. "
                "Move it aside and prepare again."
            )
        print(f"[skip] pinned checkout already exists: {target}")
        return
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"Existing target is not a Git checkout: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", url, str(target)])
    run(["git", "checkout", revision], cwd=target)


def snapshot_download(repo_id: str, target: Path, revision: str | None = None) -> None:
    if target.exists() and any(target.iterdir()):
        print(f"[skip] already exists: {target}")
        return
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before downloading datasets.") from exc
    target.mkdir(parents=True, exist_ok=True)
    print(f"+ huggingface snapshot {repo_id} -> {target}", flush=True)
    try:
        hf_snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=target,
        )
    except Exception as exc:
        if repo_id == "nnheui/llm-srbench":
            raise SystemExit(
                "LLMSRBench requires accepting its Hugging Face access terms "
                "and logging in (`hf auth login` or HF_TOKEN)."
            ) from exc
        raise


def prepare_standard() -> None:
    snapshot_download(
        "nnheui/llm-srbench",
        EXTERNAL / "llmsrbench",
        LLMSRBENCH_REV,
    )
    clone_at("https://github.com/cavalab/srbench.git", EXTERNAL / "srbench", SRBENCH_REV)
    clone_at(
        "https://github.com/omron-sinicx/srsd-benchmark.git",
        EXTERNAL / "srsd-benchmark",
        SRSD_REV,
    )


def prepare_noise() -> None:
    target = GENERATED / "noise_sr_20"
    if (target / "manifest.csv").is_file():
        print(f"[skip] already exists: {target}")
        return
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_noise_robustness_dataset.py"),
            "--out-dir",
            str(target),
            "--formula-count",
            "20",
            "--noise-levels",
            "0,0.001,0.01",
            "--repeat-seeds",
            "3",
            "--n-train",
            "512",
            "--n-val",
            "256",
            "--n-test",
            "1024",
            "--random-state",
            "42",
        ]
    )


def prepare_interference() -> None:
    target = GENERATED / "balanced_interference_96"
    if (target / "manifest.csv").is_file():
        print(f"[skip] already exists: {target}")
        return
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_v11_balanced_ablation_dataset.py"),
            "--out-dir",
            str(target),
            "--num-cases",
            "96",
            "--n-train",
            "128",
            "--n-val",
            "128",
            "--n-test",
            "1024",
        ]
    )


def prepare_sft() -> None:
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_sft_corpus.py"),
            "unpack",
            str(ROOT / "data" / "proposer_sft"),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=["standard", "noise", "interference", "sft", "all"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions = {
        "standard": prepare_standard,
        "noise": prepare_noise,
        "interference": prepare_interference,
        "sft": prepare_sft,
    }
    selected = list(actions) if args.dataset == "all" else [args.dataset]
    os.chdir(ROOT)
    for name in selected:
        print(f"\n[data] preparing {name}")
        actions[name]()


if __name__ == "__main__":
    main()
