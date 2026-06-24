# tools/algebraic_simplify_tool.py
from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import List, Optional

import sympy as sp

from .template_fill_tool import FitResult


@dataclass
class SimplifyResult:
    """
    化简后的结果。
    """
    original_expression: str
    fitted_expression: str
    simplified_expression: str
    success: bool
    error_message: Optional[str] = None
    fit_result: Optional[FitResult] = None


class AlgebraicSimplifyTool:
    """
    对拟合后的表达式做符号化简。

    注意：
    - 这里主要化简 fitted_expression，而不是原始模板 expression
    - 因为 fitted_expression 已经把参数替换成数值，更利于去重
    """

    MAX_SIMPLIFY_EXPR_LEN = max(80, int(os.environ.get("LLMSR_SIMPLIFY_MAX_EXPR_LEN", "240")))
    HUGE_CONSTANT_PATTERN = re.compile(r"(?:\d+\.\d+e[+-]?\d{3,}|\d{8,})")

    def simplify_single(self, fit_result: FitResult) -> SimplifyResult:
        """
        化简单个 FitResult。
        """
        if not fit_result.success:
            return SimplifyResult(
                original_expression=fit_result.expression,
                fitted_expression=fit_result.fitted_expression,
                simplified_expression=fit_result.fitted_expression,
                success=False,
                error_message=fit_result.error_message,
                fit_result=fit_result,
            )

        try:
            fitted_expression = str(fit_result.fitted_expression or "")
            if (
                len(fitted_expression) > self.MAX_SIMPLIFY_EXPR_LEN
                or self.HUGE_CONSTANT_PATTERN.search(fitted_expression)
            ):
                return SimplifyResult(
                    original_expression=fit_result.expression,
                    fitted_expression=fitted_expression,
                    simplified_expression=fitted_expression,
                    success=True,
                    error_message=None,
                    fit_result=fit_result,
                )

            expr = sp.sympify(fitted_expression)

            # 第一版先用 simplify 即可
            simplified = sp.simplify(expr)

            return SimplifyResult(
                original_expression=fit_result.expression,
                fitted_expression=fit_result.fitted_expression,
                simplified_expression=str(simplified),
                success=True,
                error_message=None,
                fit_result=fit_result,
            )
        except Exception as e:
            return SimplifyResult(
                original_expression=fit_result.expression,
                fitted_expression=fit_result.fitted_expression,
                simplified_expression=fit_result.fitted_expression,
                success=False,
                error_message=str(e),
                fit_result=fit_result,
            )

    def run(self, fit_results: List[FitResult]) -> List[SimplifyResult]:
        """
        批量化简。
        """
        return [self.simplify_single(r) for r in fit_results]
