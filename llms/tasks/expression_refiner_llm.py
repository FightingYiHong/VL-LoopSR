# llms/tasks/expression_refiner_llm.py
from __future__ import annotations

from typing import Dict, List, Optional

from ..llm_client import LLMClient
from ..prompt_builder import PromptBuilder
from ..response_parser import ResponseParser


class ExpressionRefinerLLM:
    """
    ExpressionRefinerLLM 负责对已有表达式进行局部改写、扩展或补全。

    它一般输入：
    - 当前较优表达式
    - 残差模式
    - 物理检查结果
    - 结构线索

    输出：
    {
        "refined_candidates": [
            {
                "expression": "...",
                "edit_type": "add_term",
                "based_on": "...",
                "rationale": "..."
            }
        ]
    }

    使用建议：
    - 优先做局部修改，而不是重写整个表达式
    - 更适合在第 2 轮及以后调用
    """

    def __init__(
        self,
        client: LLMClient,
        prompt_builder: Optional[PromptBuilder] = None,
        response_parser: Optional[ResponseParser] = None,
    ):
        self.client = client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_parser = response_parser or ResponseParser()

    def refine(
        self,
        base_expressions: List[str],
        residual_hints: Optional[List[str]] = None,
        physics_hints: Optional[List[str]] = None,
        structure_hints: Optional[List[str]] = None,
        num_candidates: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 0.95,
    ) -> Dict:
        """
        对当前表达式进行 refine。

        参数说明：
        - base_expressions:
            当前准备被 refine 的表达式列表
        - residual_hints:
            残差线索，例如：
            ["residual grows with x1", "underfit at large x0"]
        - physics_hints:
            物理/合法性线索，例如：
            ["unit mismatch risk", "singularity near x1=0"]
        - structure_hints:
            结构线索，例如：
            ["possible additive correction", "try denominator correction"]
        - num_candidates:
            返回多少个 refined 候选

        返回：
        {
            "refined_candidates": [...],
            "raw_text": "...",
            "messages": [...],
        }
        """
        messages = self.prompt_builder.build_refiner_messages(
            base_expressions=base_expressions,
            residual_hints=residual_hints,
            physics_hints=physics_hints,
            structure_hints=structure_hints,
            num_candidates=num_candidates,
        )

        response = self.client.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        parsed = self.response_parser.parse_refiner_response(response.text)

        return {
            "refined_candidates": parsed.get("refined_candidates", []),
            "raw_text": response.text,
            "messages": messages,
        }

    def refine_from_summary(
        self,
        summary_text: str,
        num_candidates: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 0.95,
    ) -> Dict:
        """
        一个更灵活的辅助接口。
        当上层已经把任务状态总结成一段文本时，可直接使用。
        """
        system_prompt = (
            "You are an expert symbolic regression expression refiner. "
            "Return ONLY valid JSON."
        )

        user_prompt = f"""
Given the following symbolic regression refinement summary, produce up to {num_candidates}
locally improved candidate expressions.

Summary:
{summary_text}

Return JSON in this format:
{{
  "refined_candidates": [
    {{
      "expression": "a * x0 / (b + x1) + c",
      "edit_type": "add_term",
      "based_on": "a * x0 / (b + x1)",
      "rationale": "brief reason"
    }}
  ]
}}
""".strip()

        messages = self.prompt_builder.build_plain_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        response = self.client.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        parsed = self.response_parser.parse_refiner_response(response.text)

        return {
            "refined_candidates": parsed.get("refined_candidates", []),
            "raw_text": response.text,
            "messages": messages,
        }