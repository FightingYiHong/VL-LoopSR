# tools/scoring_tool.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import sympy as sp

from .algebraic_simplify_tool import SimplifyResult


@dataclass
class ScoreResult:
    """
    最终用于排序的结果。
    """
    expression: str
    simplified_expression: str

    train_mse: Optional[float]
    val_mse: Optional[float]
    test_mse: Optional[float]

    complexity: int
    score: float

    success: bool
    error_message: Optional[str] = None

    source: Optional[SimplifyResult] = None


class ScoringTool:
    """
    对候选表达式进行排序打分。

    第一版先用一个很直接的规则：
        score = val_mse + complexity_weight * complexity

    其中：
    - val_mse 越低越好
    - complexity 越低越好
    - score 越低越好
    """

    def __init__(self, complexity_weight=None):
        self.complexity_weight = complexity_weight

    @staticmethod
    def _estimate_complexity(expr_str: str) -> int:
        """
        用 sympy 表达式树节点数估计复杂度。
        """
        try:
            expr = sp.sympify(expr_str)
            return sum(1 for _ in sp.preorder_traversal(expr))
        except Exception:
            # 解析失败时给个较大复杂度
            return 9999

    def score_single(self, item: SimplifyResult) -> ScoreResult:
        """
        给单个化简结果打分。
        """
        fit_result = item.fit_result

        if fit_result is None or fit_result.val_mse is None:
            return ScoreResult(
                expression=item.original_expression,
                simplified_expression=item.simplified_expression,
                train_mse=None,
                val_mse=None,
                test_mse=None,
                complexity=9999,
                score=float("inf"),
                success=False,
                error_message=item.error_message or "fit_result 缺失",
                source=item,
            )

        complexity = self._estimate_complexity(item.simplified_expression)
        if self.complexity_weight is None:
            score = float(fit_result.val_mse)
        else:
         score = float(fit_result.val_mse) + self.complexity_weight * complexity


        return ScoreResult(
            expression=item.original_expression,
            simplified_expression=item.simplified_expression,
            train_mse=fit_result.train_mse,
            val_mse=fit_result.val_mse,
            test_mse=fit_result.test_mse,
            complexity=complexity,
            score=score,
            success=fit_result.success,
            error_message=item.error_message,
            source=item,
        )

    def run(self, simplify_results: List[SimplifyResult]) -> List[ScoreResult]:
        """
        批量打分并排序。
        """
        scored = [self.score_single(item) for item in simplify_results]
        scored.sort(key=lambda x: x.score)
        return scored