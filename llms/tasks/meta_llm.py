# llms/tasks/meta_llm.py
from __future__ import annotations

from typing import Dict, List, Optional

from ..llm_client import LLMClient
from ..prompt_builder import PromptBuilder
from ..response_parser import ResponseParser


class MetaLLM:
    """
    MetaLLM 负责做“控制决策”：
    当前 top-k 表达式出来后，判断是否还需要继续 refine。

    它通常不会直接生成表达式，而是输出一个控制信号，比如：
    {
        "refine": true,
        "reason": "...",
        "focus": [...],
        "strategy": "local_rewrite"
    }

    在整个 pipeline 中，它更像一个“策略控制器”。
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

    def decide(
        self,
        top_expressions: List[str],
        score_summary: Optional[List[str]] = None,
        residual_hints: Optional[List[str]] = None,
        physics_hints: Optional[List[str]] = None,
        stop_threshold_hint: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ) -> Dict:
        """
        根据当前 top 表达式和误差信息，判断是否需要继续 refine。

        参数说明：
        - top_expressions:
            当前 top-k 表达式列表
        - score_summary:
            打分摘要，比如：
            ["expr1 val_mse=0.01 complexity=5", "expr2 val_mse=0.015 complexity=4"]
        - residual_hints:
            残差模式摘要，比如：
            ["residual still correlated with x1", "underestimation at large x0"]
        - physics_hints:
            物理检查摘要，比如：
            ["unit consistent", "has singularity near x1=-b"]
        - stop_threshold_hint:
            可以补充一句规则，例如：
            "stop if val_mse < 1e-4 and expression is simple"

        返回：
        {
            "refine": bool,
            "reason": str,
            "focus": list[str],
            "strategy": str,
            "raw_text": str,
            "messages": list[Message],
        }
        """
        messages = self.prompt_builder.build_meta_messages(
            top_expressions=top_expressions,
            score_summary=score_summary,
            residual_hints=residual_hints,
            physics_hints=physics_hints,
            stop_threshold_hint=stop_threshold_hint,
        )

        response = self.client.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        parsed = self.response_parser.parse_meta_response(response.text)

        parsed["raw_text"] = response.text
        parsed["messages"] = messages
        return parsed

    def decide_from_summary(
        self,
        summary_text: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ) -> Dict:
        """
        一个更灵活的辅助接口。
        当你已经把当前状态整理成一段文本时，可以直接让 MetaLLM 做决策。
        """
        system_prompt = (
            "You are an expert symbolic regression controller. "
            "Return ONLY valid JSON."
        )

        user_prompt = f"""
Given the following symbolic regression status summary, decide whether refinement is necessary.

Summary:
{summary_text}

Return JSON in this format:
{{
  "refine": true,
  "reason": "brief reason",
  "focus": ["actionable item 1", "actionable item 2"],
  "strategy": "local_rewrite"
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

        parsed = self.response_parser.parse_meta_response(response.text)
        parsed["raw_text"] = response.text
        parsed["messages"] = messages
        return parsed