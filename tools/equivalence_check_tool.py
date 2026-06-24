# tools/equivalence_check_tool.py
from __future__ import annotations

from typing import Dict, List

from .algebraic_simplify_tool import SimplifyResult


class EquivalenceCheckTool:
    """
    等价表达式去重工具。

    第一版策略：
    - 用 simplified_expression 作为 canonical key
    - 同一个 key 只保留验证误差更低的那个
    """

    @staticmethod
    def _score_for_keep(item: SimplifyResult) -> float:
        """
        返回一个用于比较的分数。
        分数越小越好。
        """
        fit_result = item.fit_result
        if fit_result is None or fit_result.val_mse is None:
            return float("inf")
        return fit_result.val_mse

    def deduplicate(self, simplify_results: List[SimplifyResult]) -> List[SimplifyResult]:
        """
        去重并保留每组中更好的一个。
        """
        best_by_expr: Dict[str, SimplifyResult] = {}

        for item in simplify_results:
            key = item.simplified_expression

            if key not in best_by_expr:
                best_by_expr[key] = item
            else:
                current_best = best_by_expr[key]
                if self._score_for_keep(item) < self._score_for_keep(current_best):
                    best_by_expr[key] = item

        return list(best_by_expr.values())

    def run(self, simplify_results: List[SimplifyResult]) -> List[SimplifyResult]:
        return self.deduplicate(simplify_results)