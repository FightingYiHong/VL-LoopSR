import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import test_fey  # noqa: E402
from tools.template_fill_tool import FitResult


def make_fit(expression, val_mse, success=True):
    return FitResult(
        expression=expression,
        fitted_expression=expression,
        parameters={},
        train_mse=val_mse,
        val_mse=val_mse,
        test_mse=None,
        success=success,
    )


def test_candidate_search_audit_preserves_order_and_unique_counts():
    dataset = SimpleNamespace(
        val_df=pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 2.0]}),
        target_name="y",
    )
    test_fey.reset_candidate_search_audit(dataset)
    test_fey.record_candidate_search_audit(
        dataset,
        "initial_prefilter",
        [make_fit("x", 0.01), make_fit("x + 1", 0.0005)],
    )
    test_fey.record_candidate_search_audit(
        dataset,
        "loop_initial",
        [make_fit("x", 0.01), make_fit("2*x", 0.0)],
    )

    records = test_fey.candidate_search_audit_state(dataset)["records"]
    assert [item["evaluation_index"] for item in records] == [1, 2, 3, 4]
    assert [item["unique_evaluations_seen"] for item in records] == [1, 2, 2, 3]
    assert records[2]["is_new_unique_candidate"] is False
    assert records[3]["val_r2"] == 1.0


def test_candidate_search_audit_can_be_shared_with_prefilter_dataset():
    source = SimpleNamespace(val_df=pd.DataFrame({"y": [0.0, 1.0]}), target_name="y")
    target = SimpleNamespace(val_df=pd.DataFrame({"y": [0.0, 1.0]}), target_name="y")
    test_fey.reset_candidate_search_audit(source)
    test_fey.share_candidate_search_audit(source, target)
    test_fey.record_candidate_search_audit(target, "prefilter", [make_fit("x", 0.0)])

    assert len(test_fey.candidate_search_audit_state(source)["records"]) == 1
