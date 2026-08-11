from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_coverage_uses_first_held_out_hit(tmp_path: Path) -> None:
    candidates = pd.DataFrame(
        [
            {"method": "A", "task_id": "t1", "evaluation_index": 1, "test_r2": 0.5},
            {"method": "A", "task_id": "t1", "evaluation_index": 2, "test_r2": 0.9995},
            {"method": "A", "task_id": "t2", "evaluation_index": 1, "test_r2": 0.1},
            {"method": "B", "task_id": "t1", "evaluation_index": 3, "test_r2": 1.0},
            {"method": "B", "task_id": "t2", "evaluation_index": 1, "test_r2": 0.999},
        ]
    )
    finals = pd.DataFrame(
        [
            {"method": "A", "task_id": "t1", "test_r2": 0.9995},
            {"method": "A", "task_id": "t2", "test_r2": 0.1},
            {"method": "B", "task_id": "t1", "test_r2": 0.2},
            {"method": "B", "task_id": "t2", "test_r2": 1.0},
        ]
    )
    candidate_path = tmp_path / "candidates.csv"
    final_path = tmp_path / "finals.csv"
    output_dir = tmp_path / "out"
    candidates.to_csv(candidate_path, index=False)
    finals.to_csv(final_path, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_candidate_coverage.py"),
            "--candidates-csv",
            str(candidate_path),
            "--finals-csv",
            str(final_path),
            "--output-dir",
            str(output_dir),
            "--methods",
            "A,B",
            "--expected-tasks",
            "2",
            "--max-budget",
            "3",
        ],
        check=True,
    )

    curve = pd.read_csv(output_dir / "candidate_coverage_curve.csv")
    summary = pd.read_csv(output_dir / "candidate_coverage_summary.csv")
    assert curve.query("method == 'A' and candidate_budget == 1")["tasks_solved"].item() == 0
    assert curve.query("method == 'A' and candidate_budget == 2")["tasks_solved"].item() == 1
    assert curve.query("method == 'B' and candidate_budget == 3")["tasks_solved"].item() == 1
    assert summary.set_index("method").loc["A", "final_selected_expression"] == 1
    assert summary.set_index("method").loc["B", "final_selected_expression"] == 1
