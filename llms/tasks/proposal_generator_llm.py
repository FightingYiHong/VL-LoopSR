# llms/tasks/proposal_generator_llm.py
from __future__ import annotations

from typing import Dict, List, Optional
from llms.types import Message
from ..llm_client import LLMClient
from ..prompt_builder import PromptBuilder
from ..response_parser import ResponseParser


class ProposalGeneratorLLM:
    """
    ProposalGeneratorLLM 负责生成符号回归候选表达式。

    它通常位于 pipeline 的前半部分，输入包括：
    - 变量名
    - target 名
    - 结构线索（单调性、可分离性、对称性等）
    - 视觉线索（如果你做了 plot + visual reasoning）
    - 单位线索
    - 运算符白名单

    输出是一个结构化 dict：
    {
        "candidates": [
            {
                "expression": "...",
                "skeleton": "...",
                "parameters": [...],
                "rationale": "...",
                "prior_score": 0.0
            }
        ]
    }

    设计目标：
    - 不让上层直接手写 prompt
    - 不让上层直接解析模型返回
    - 出错时也尽量返回稳定结构
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

    def generate_candidates(
        self,
        variable_names: List[str],
        target_name: str = "y",
        structure_hints: Optional[List[str]] = None,
        visual_hints: Optional[List[str]] = None,
        unit_hints: Optional[Dict[str, str]] = None,
        allowed_operators: Optional[List[str]] = None,
        num_candidates: int = 5,
        extra_constraints: Optional[List[str]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 0.95,
    ) -> Dict:
        """
        生成候选表达式。

        参数说明：
        - variable_names:
            输入变量名，例如 ["x0", "x1"]
        - target_name:
            目标变量名，默认是 "y"
        - structure_hints:
            结构线索，例如：
            ["y is monotonic increasing in x0", "possible multiplicative interaction"]
        - visual_hints:
            来自图像分析或 visual_reasoning_llm 的线索
        - unit_hints:
            单位信息，例如 {"x0": "m", "x1": "s", "y": "m/s"}
        - allowed_operators:
            运算符白名单，例如 ["+", "-", "*", "/", "log", "exp"]
        - num_candidates:
            希望返回多少个候选
        - extra_constraints:
            额外约束，例如 ["prefer simple expressions", "avoid trigonometric functions"]

        返回值：
        {
            "candidates": [...],
            "raw_text": "...",
            "messages": [...],
        }

        这样设计的原因是：
        - "candidates" 供后续工具直接使用
        - "raw_text" 方便调试 LLM 原始输出
        - "messages" 方便复现 prompt
        """
        messages = self.prompt_builder.build_proposal_messages(
            variable_names=variable_names,
            target_name=target_name,
            structure_hints=structure_hints,
            visual_hints=visual_hints,
            unit_hints=unit_hints,
            allowed_operators=allowed_operators,
            num_candidates=num_candidates,
            extra_constraints=extra_constraints,
        )

        response = self.client.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        parsed = self.response_parser.parse_proposal_response(response.text)

        return {
            "candidates": parsed.get("candidates", []),
            "raw_text": response.text,
            "messages": messages,
        }

    def generate_candidates_from_summary(
        self,
        summary_text: str,
        num_candidates: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 0.95,
    ) -> Dict:
        """
        一个更灵活的辅助接口。
        当你上层已经有一段整理好的任务摘要时，可以直接用这个方法。

        例如：
        summary_text 可以是你自己拼好的数据摘要、结构线索和约束。

        这种方式适合：
        - 你还没把结构化输入完全定好
        - 想先快速实验 prompt 效果
        """
        system_prompt = (
            "You are an expert symbolic regression assistant. "
            "Return ONLY valid JSON with a top-level field named 'candidates'."
        )

        user_prompt = f"""
Given the following task summary, propose up to {num_candidates} candidate symbolic regression expressions.

Task summary:
{summary_text}

Return JSON with this format:
{{
  "candidates": [
    {{
      "expression": "a * x0 / (b + x1)",
      "skeleton": "x0 / (const + x1)",
      "parameters": ["a", "b"],
      "rationale": "brief reason",
      "prior_score": 0.8
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

        parsed = self.response_parser.parse_proposal_response(response.text)

        return {
            "candidates": parsed.get("candidates", []),
            "raw_text": response.text,
            "messages": messages,
        }

    def _make_csv_points_text(self, df, variable_names: List[str], target_name: str, max_rows: int = 20) -> str:
        """
        把 dataframe 前若干行转成 prompt 里的 CSV 文本。
        """
        cols = variable_names + [target_name]
        sub_df = df[cols].head(max_rows).copy()
        return sub_df.to_csv(index=False).strip()

    def generate_single_expression_from_data(
        self,
        df,
        variable_names: List[str],
        target_name: str = "y",
        allowed_operators: Optional[List[str]] = None,
        max_rows: int = 20,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ) -> Dict:
        """
        直接根据数据点生成一个闭式表达式。
        这是新的首选 proposal 模式。
        """
        csv_points = self._make_csv_points_text(
            df=df,
            variable_names=variable_names,
            target_name=target_name,
            max_rows=max_rows,
        )

        messages = self.prompt_builder.build_single_expression_from_data_messages(
            csv_points=csv_points,
            variable_names=variable_names,
            target_name=target_name,
            allowed_operators=allowed_operators,
        )

        response = self.client.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        parsed = self.response_parser.parse_answer_tag_expression(
            response.text,
            target_name=target_name,
        )

        expr = parsed.get("expression", "").strip()

        candidates = []
        if expr:
            candidates.append(
                {
                    "expression": expr,
                    "skeleton": "",
                    "parameters": [],
                    "rationale": "single-expression proposal from data points",
                    "prior_score": 1.0,
                }
            )

        return {
            "candidates": candidates,
            "raw_text": response.text,
            "messages": messages,
        }

    def generate_multiple_single_expressions(
        self,
        df,
        variable_names,
        target_name: str = "y",
        allowed_operators=None,
        max_rows: int = 20,
        num_calls: int = 10,
        temperatures=None,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ):
        """
        多次调用单表达式提案器，收集多个候选。

        设计目的：
        - 避免一次长输出导致截断
        - 利用多次短调用提升候选多样性
        """
        if temperatures is None:
            temperatures = [0.1, 0.2, 0.3, 0.4, 0.5]

        all_candidates = []
        raw_texts = []

        for i in range(num_calls):
            temp = temperatures[i % len(temperatures)]

            result = self.generate_single_expression_from_data(
                df=df,
                variable_names=variable_names,
                target_name=target_name,
                allowed_operators=allowed_operators,
                max_rows=max_rows,
                temperature=temp,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            raw_texts.append(result.get("raw_text", ""))

            for item in result.get("candidates", []):
                if not isinstance(item, dict):
                    continue
                expr = str(item.get("expression", "")).strip()
                if expr:
                    all_candidates.append(
                        {
                            "expression": expr,
                            "skeleton": "",
                            "parameters": [],
                            "rationale": f"multi-call single-expression proposal, call={i+1}, temp={temp}",
                            "prior_score": 1.0,
                        }
                    )

        # 去重
        unique_candidates = []
        seen = set()
        for item in all_candidates:
            expr = item["expression"]
            if expr not in seen:
                unique_candidates.append(item)
                seen.add(expr)

        return {
            "candidates": unique_candidates,
            "raw_text": "\n\n".join(raw_texts),
            "messages": [],
        }


    def _make_csv_points_text(self, df, variable_names, target_name: str, max_rows: int = 20) -> str:
        """
        把 dataframe 前若干行转成 CSV 文本，供 prompt 使用。
        """
        cols = variable_names + [target_name]
        sub_df = df[cols].head(max_rows).copy()
        return sub_df.to_csv(index=False).strip()


    def generate_single_expression_from_plot_and_data(
        self,
        df,
        variable_names,
        target_name: str = "y",
        image_paths=None,
        allowed_operators=None,
        max_rows: int = 20,
        temperature: float = 0.1,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ):
        """
        结合散点图和数据点，生成一个单表达式候选。
        """
        if image_paths is None:
            image_paths = []

        csv_points = self._make_csv_points_text(
            df=df,
            variable_names=variable_names,
            target_name=target_name,
            max_rows=max_rows,
        )

        messages = self.prompt_builder.build_multimodal_single_expression_messages(
            csv_points=csv_points,
            variable_names=variable_names,
            target_name=target_name,
            image_paths=image_paths,
            allowed_operators=allowed_operators,
        )

        response = self.client.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        parsed = self.response_parser.parse_answer_tag_expression(
            response.text,
            target_name=target_name,
        )

        expr = parsed.get("expression", "").strip()

        candidates = []
        if expr:
            candidates.append(
                {
                    "expression": expr,
                    "skeleton": "",
                    "parameters": [],
                    "rationale": "single-expression proposal from plot and data",
                    "prior_score": 1.0,
                }
            )

        return {
            "candidates": candidates,
            "raw_text": response.text,
            "messages": messages,
        }

    def _make_csv_points_text(self, df, variable_names, target_name: str, max_rows: int = 20) -> str:
        """
        把 dataframe 前若干行转成 CSV 文本，供 prompt 使用。
        """
        cols = variable_names + [target_name]
        sub_df = df[cols].head(max_rows).copy()
        return sub_df.to_csv(index=False).strip()


    def _build_dimension_aware_text_prompt(
        self,
        variable_names,
        target_name: str,
        csv_points: str,
        plot_descriptions=None,
        allowed_operators=None,
    ):
        """
        根据输入维度自动构造更合适的 text prompt。
        """
        if allowed_operators is None:
            allowed_operators = [
                "+", "-", "*", "/", "**",
                "sqrt", "sin", "cos", "tan",
                "exp", "log", "abs", "pi", "e"
            ]

        d = len(variable_names)
        vars_text = ", ".join(variable_names)
        ops_text = ", ".join(allowed_operators)

        if plot_descriptions is None:
            plot_descriptions = []

        if d == 1:
            structure_hint = """
    Consider:
    - affine / linear forms
    - polynomial and power forms
    - rational forms
    - trigonometric and oscillatory forms
    - composite nonlinear forms such as sin(x1**2), cos(x1**2), sin(x1)*cos(x1)
    """.strip()
        elif d == 2:
            structure_hint = """
    Consider:
    - additive forms: f(x1) + g(x2)
    - multiplicative interactions: x1 * x2
    - low-order polynomial interactions
    - separable or partially separable structure
    - trigonometric interaction forms such as sin(x1)*cos(x2)
    - rational corrections involving x1 and x2
    """.strip()
        else:
            structure_hint = """
    Consider:
    - sparse low-order interactions
    - additive or partially separable structure
    - only a few important variables may dominate
    - avoid overly dense formulas involving all variables at once
    - start from simple low-order interaction structure
    """.strip()

        plot_text = "\n".join(f"- {x}" for x in plot_descriptions) if plot_descriptions else "- no plot descriptions available"

        prompt = f"""
    You are doing Symbolic Regression.

    Variables:
    {vars_text}
    Target:
    {target_name}

    Available operators:
    {ops_text}

    Plot descriptions:
    {plot_text}

    Structure guidance:
    {structure_hint}

    Task:
    Infer one plausible symbolic expression:
    {target_name} = f({vars_text})

    Rules:
    1) Output ONLY one line:
    <ANSWER>{target_name} = ...</ANSWER>

    2) You may use free parameters such as a, b, c, d if needed.
    3) Use variable names exactly as given.
    4) Prefer a structurally meaningful expression over a poor trivial affine fit.
    5) Do not output JSON.
    6) Do not explain.

    <Data>
    {csv_points}
    </Data>
    """.strip()

        return prompt


    def generate_dimension_aware_single_expression(
        self,
        df,
        variable_names,
        target_name: str = "y",
        plot_descriptions=None,
        allowed_operators=None,
        max_rows: int = 20,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ):
        """
        根据输入维度自动构造 prompt，生成一个表达式候选。
        这是 text-only 版本，适合 1D / 2D / 高维通用使用。
        """
        csv_points = self._make_csv_points_text(
            df=df,
            variable_names=variable_names,
            target_name=target_name,
            max_rows=max_rows,
        )

        system_prompt = (
            "You are an expert symbolic regression assistant. "
            "Infer a mathematically meaningful expression from data and plot summaries. "
            "Return only the final answer in the required format."
        )

        user_prompt = self._build_dimension_aware_text_prompt(
            variable_names=variable_names,
            target_name=target_name,
            csv_points=csv_points,
            plot_descriptions=plot_descriptions,
            allowed_operators=allowed_operators,
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        response = self.client.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        parsed = self.response_parser.parse_answer_tag_expression(
            response.text,
            target_name=target_name,
        )

        expr = parsed.get("expression", "").strip()

        candidates = []
        if expr:
            candidates.append(
                {
                    "expression": expr,
                    "skeleton": "",
                    "parameters": [],
                    "rationale": "dimension-aware single-expression proposal",
                    "prior_score": 1.0,
                }
            )

        return {
            "candidates": candidates,
            "raw_text": response.text,
            "messages": messages,
        }


    def generate_multiple_dimension_aware_single_expressions(
        self,
        df,
        variable_names,
        target_name: str = "y",
        plot_descriptions=None,
        allowed_operators=None,
        max_rows: int = 20,
        num_calls: int = 10,
        temperatures=None,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ):
        """
        多次调用 dimension-aware 单表达式提案器，收集多个候选。
        """
        if temperatures is None:
            temperatures = [0.1, 0.2, 0.3, 0.4, 0.5]

        all_candidates = []
        raw_texts = []

        for i in range(num_calls):
            temp = temperatures[i % len(temperatures)]

            result = self.generate_dimension_aware_single_expression(
                df=df,
                variable_names=variable_names,
                target_name=target_name,
                plot_descriptions=plot_descriptions,
                allowed_operators=allowed_operators,
                max_rows=max_rows,
                temperature=temp,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            raw_texts.append(result.get("raw_text", ""))

            for item in result.get("candidates", []):
                if not isinstance(item, dict):
                    continue
                expr = str(item.get("expression", "")).strip()
                if expr:
                    all_candidates.append(
                        {
                            "expression": expr,
                            "skeleton": "",
                            "parameters": [],
                            "rationale": f"dimension-aware multi-call proposal, call={i+1}, temp={temp}",
                            "prior_score": 1.0,
                        }
                    )

        unique_candidates = []
        seen = set()
        for item in all_candidates:
            expr = item["expression"]
            if expr not in seen:
                unique_candidates.append(item)
                seen.add(expr)

        return {
            "candidates": unique_candidates,
            "raw_text": "\n\n".join(raw_texts),
            "messages": [],
        }