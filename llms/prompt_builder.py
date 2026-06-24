# llms/prompt_builder.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from llms.types import Message
from .types import Message


class PromptBuilder:
    """
    PromptBuilder 负责把“任务输入”组织成统一的 prompt。

    设计目标：
    1. 不让各个 task 文件自己拼接长字符串
    2. 让 prompt 模板更可复用、更容易维护
    3. 让输出格式要求（尤其是 JSON 输出）集中管理

    使用方式示例：
        pb = PromptBuilder()

        messages = pb.build_proposal_messages(
            variable_names=["x0", "x1"],
            structure_hints=["x0 monotonic increasing", "possible multiplicative interaction"],
            visual_hints=["curve looks saturating"],
            num_candidates=5,
        )

    然后把 messages 丢给 LLMClient.generate(...) 即可。
    """

    # =========================
    # 通用 system prompt 模板
    # =========================
    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert symbolic regression assistant. "
        "Your job is to propose, analyze, and refine mathematical expressions "
        "that predict a target variable from input variables. "
        "You must follow the requested output format exactly."
    )

    STRICT_JSON_RULES = (
        "Return ONLY valid JSON. "
        "Do not include markdown code fences. "
        "Do not include explanations before or after the JSON."
    )

    SAFE_EXPR_RULES = (
        "Use only simple math operators and functions unless explicitly allowed: "
        "+, -, *, /, pow, sin, cos, exp, log, sqrt. "
        "Use variable names exactly as given. "
        "Do not invent new variables."
    )

    # =========================
    # 基础工具函数
    # =========================
    @staticmethod
    def _format_list(items: Optional[Sequence[str]], default: str = "None") -> str:
        """
        把字符串列表格式化成可读文本。
        """
        if not items:
            return default
        return "\n".join(f"- {item}" for item in items)

    def build_single_expression_from_data_messages(
        self,
        csv_points: str,
        variable_names: List[str],
        target_name: str = "y",
        allowed_operators: Optional[List[str]] = None,
    ) -> List[Message]:
        """
        直接从数据点推断一个闭式表达式。
        这是新的主提案模式，优先于旧的 JSON 多候选模式。
        """
        system_prompt = (
            "You are an expert symbolic regression assistant. "
            "Infer a closed-form mathematical expression from exact numeric data. "
            "Return only the final answer in the required format."
        )

        var_text = ", ".join(variable_names + [target_name])
        op_text = ", ".join(allowed_operators) if allowed_operators else "+, -, *, /, **, sqrt, sin, cos, tan, exp, log, abs, pi, e"

        user_prompt = f"""
    You are doing Symbolic Regression.

    You are given a set of exact numeric data points in <DATA> with columns:
    {var_text}

    Your task:
    Infer one closed-form symbolic expression {target_name} = f({", ".join(variable_names)})
    that best matches the data.

    Allowed building blocks:
    {op_text}

    Rules (STRICT):
    1) Output ONLY one line:
    <ANSWER>{target_name} = ...</ANSWER>

    2) Do NOT use free parameters like a, b, c, k, m, n.
    Use only explicit constants.

    3) Use variable names exactly as given:
    {", ".join(variable_names)}

    4) Prefer simple exact forms when possible:
    - rational constants
    - roots
    - pi and e
    - trigonometric expressions if supported by data
    - polynomial, rational, exponential, logarithmic, or trigonometric forms when appropriate

    5) If needed, prefer a nonlinear exact expression over a poor linear approximation.

    6) Do not explain. Do not output JSON.

    <DATA>
    {csv_points}
    </DATA>
    """.strip()

        return self._make_messages(system_prompt, user_prompt)

    @staticmethod
    def _format_kv_dict(data: Optional[Dict], default: str = "None") -> str:
        """
        把 dict 格式化成多行 key: value。
        """
        if not data:
            return default
        lines = []
        for k, v in data.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    @staticmethod
    def _make_messages(system_prompt: str, user_prompt: str) -> List[Message]:
        """
        统一转成 chat message 格式。
        """
        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

    # =========================
    # JSON schema 文本模板
    # =========================
    @staticmethod
    def proposal_output_schema_text(num_candidates: int) -> str:
        """
        给 proposal generator 使用的 JSON 输出格式说明。
        这里故意写成“文本 schema”，方便塞进 prompt，而不是要求真正 JSON Schema。
        """
        return f"""
Expected JSON format:
{{
  "candidates": [
    {{
      "expression": "a * x0 / (b + x1)",
      "skeleton": "x0 / (const + x1)",
      "parameters": ["a", "b"],
      "rationale": "brief reason",
      "prior_score": 0.78
    }}
  ]
}}

Requirements:
- Return at most {num_candidates} candidates.
- "expression" must be a valid symbolic expression string.
- "skeleton" should be a simplified structural template.
- "parameters" should contain only free constant names such as ["a", "b", "c"].
- "prior_score" should be a float in [0, 1].
""".strip()

    @staticmethod
    def meta_output_schema_text() -> str:
        """
        给 meta controller 使用的输出格式说明。
        """
        return """
Expected JSON format:
{
  "refine": true,
  "reason": "brief reason",
  "focus": [
    "reduce residual correlation with x1",
    "try denominator correction"
  ],
  "strategy": "local_rewrite"
}

Requirements:
- "refine" must be true or false.
- "reason" must be short and concrete.
- "focus" should be a list of actionable guidance.
- "strategy" should be one of:
  "local_rewrite", "add_term", "transform_variable", "stop"
""".strip()

    @staticmethod
    def refiner_output_schema_text(num_candidates: int) -> str:
        """
        给 expression refiner 使用的输出格式说明。
        """
        return f"""
Expected JSON format:
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

Requirements:
- Return at most {num_candidates} refined candidates.
- "edit_type" should be one of:
  "local_rewrite", "add_term", "replace_subtree", "transform_variable"
- "based_on" must refer to the original expression being refined.
""".strip()

    # =========================
    # Proposal 任务 prompt
    # =========================
    def build_proposal_messages(
        self,
        variable_names: List[str],
        target_name: str = "y",
        structure_hints: Optional[List[str]] = None,
        visual_hints: Optional[List[str]] = None,
        unit_hints: Optional[Dict[str, str]] = None,
        allowed_operators: Optional[List[str]] = None,
        num_candidates: int = 5,
        extra_constraints: Optional[List[str]] = None,
    ) -> List[Message]:
        """
        为候选表达式生成任务构造 prompt。

        参数说明：
        - variable_names: 输入变量名，例如 ["x0", "x1"]
        - target_name: 目标变量名，默认 "y"
        - structure_hints: 结构提示，比如单调性、可分离性等
        - visual_hints: 视觉线索，比如“曲线有饱和趋势”
        - unit_hints: 单位信息
        - allowed_operators: 运算符白名单
        - num_candidates: 希望生成多少候选
        - extra_constraints: 额外约束，比如“避免 trigonometric functions”
        """
        system_prompt = (
            f"{self.DEFAULT_SYSTEM_PROMPT}\n"
            f"{self.STRICT_JSON_RULES}\n"
            f"{self.SAFE_EXPR_RULES}"
        )

        operators_text = ", ".join(allowed_operators) if allowed_operators else "not specified"

        user_prompt = f"""
Task:
Generate candidate symbolic regression expressions to predict target "{target_name}"
from input variables {variable_names}.

Variable names:
{self._format_list(variable_names)}

Target name:
- {target_name}

Structure hints:
{self._format_list(structure_hints)}

Visual hints:
{self._format_list(visual_hints)}

Unit hints:
{self._format_kv_dict(unit_hints)}

Allowed operators:
- {operators_text}

Extra constraints:
{self._format_list(extra_constraints)}

Please generate diverse but plausible candidate formulas.

Diversity requirement:
- include at least one affine or linear candidate
- include at least one polynomial or power-based candidate
- include at least one rational candidate
- include at least one trigonometric or oscillatory candidate when plausible

Do not return only linear expressions.
Prefer concise expressions, but keep structural diversity.
Avoid unnecessary complexity, but do not collapse everything to affine forms.

{self.proposal_output_schema_text(num_candidates)}
""".strip()

        return self._make_messages(system_prompt, user_prompt)

    # =========================
    # Meta 决策任务 prompt
    # =========================
    def build_meta_messages(
        self,
        top_expressions: List[str],
        score_summary: Optional[List[str]] = None,
        residual_hints: Optional[List[str]] = None,
        physics_hints: Optional[List[str]] = None,
        stop_threshold_hint: Optional[str] = None,
    ) -> List[Message]:
        """
        为 meta llm 构造 prompt。
        它的职责通常是：判断是否还要 refine。
        """
        system_prompt = (
            f"{self.DEFAULT_SYSTEM_PROMPT}\n"
            f"{self.STRICT_JSON_RULES}"
        )

        user_prompt = f"""
Task:
Decide whether symbolic regression refinement is necessary.

Current top expressions:
{self._format_list(top_expressions)}

Score summary:
{self._format_list(score_summary)}

Residual hints:
{self._format_list(residual_hints)}

Physics / validity hints:
{self._format_list(physics_hints)}

Stop threshold hint:
- {stop_threshold_hint or "not specified"}

Decision guideline:
- If current expressions are already accurate and simple, choose refine = false.
- If there is systematic residual pattern or clear missing structure, choose refine = true.
- Focus on actionable next steps.

{self.meta_output_schema_text()}
""".strip()

        return self._make_messages(system_prompt, user_prompt)

    # =========================
    # Refiner 任务 prompt
    # =========================
    def build_refiner_messages(
        self,
        base_expressions: List[str],
        residual_hints: Optional[List[str]] = None,
        physics_hints: Optional[List[str]] = None,
        structure_hints: Optional[List[str]] = None,
        num_candidates: int = 5,
    ) -> List[Message]:
        """
        为 expression refiner 构造 prompt。
        输入通常包括当前表达式和失败模式，然后让模型做局部修改。
        """
        system_prompt = (
            f"{self.DEFAULT_SYSTEM_PROMPT}\n"
            f"{self.STRICT_JSON_RULES}\n"
            "Prefer local modifications over completely unrelated new formulas."
        )

        user_prompt = f"""
Task:
Refine the current symbolic expressions by making local improvements.

Base expressions:
{self._format_list(base_expressions)}

Residual hints:
{self._format_list(residual_hints)}

Physics / validity hints:
{self._format_list(physics_hints)}

Structure hints:
{self._format_list(structure_hints)}

Refinement guideline:
- Prefer small local edits instead of rewriting everything.
- You may add a term, modify a denominator, replace a sub-expression,
  or apply a simple variable transform.
- Keep expressions interpretable.

{self.refiner_output_schema_text(num_candidates)}
""".strip()

        return self._make_messages(system_prompt, user_prompt)

    # =========================
    # 通用文本 prompt
    # =========================
    def build_plain_messages(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[Message]:
        """
        一个通用接口：
        当你临时要做测试，或者还没写专门 task 时，可以直接用这个。
        """
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        return self._make_messages(system_prompt, user_prompt)



    def build_multimodal_single_expression_messages(
        self,
        csv_points: str,
        variable_names,
        target_name: str,
        image_paths,
        allowed_operators=None,
    ):
        """
        构造多模态单表达式提案消息。

        目标：
        - 模型同时看到散点图和少量数值点
        - 输出单个闭式表达式
        """
        if allowed_operators is None:
            allowed_operators = [
                "+", "-", "*", "/", "**",
                "sqrt", "sin", "cos", "tan",
                "exp", "log", "abs", "pi", "e"
            ]

        system_prompt = (
            "You are an expert multimodal symbolic regression assistant. "
            "You can infer mathematical expressions from plots and exact data points. "
            "Return only the final answer in the required format."
        )

        var_text = ", ".join(variable_names + [target_name])
        op_text = ", ".join(allowed_operators)

        text_prompt = f"""
        You are doing Symbolic Regression.

        You are given:
        1. Scatter plot images of the dataset
        2. Exact numeric data points in <DATA>

        Columns:
        {var_text}

        Your task:
        Infer one closed-form symbolic expression {target_name} = f({", ".join(variable_names)})
        that best matches the data and the plot shape.

        Allowed building blocks:
        {op_text}

        Rules (STRICT):
        1) Output ONLY one line:
        <ANSWER>{target_name} = ...</ANSWER>

        2) Do NOT use free parameters like a, b, c, k, m, n.
        Use only explicit constants.

        3) Use variable names exactly as given:
        {", ".join(variable_names)}

        4) Prefer exact or near-exact symbolic forms when possible.

        5) If the plot suggests oscillation, periodicity, symmetry, saturation, or curvature,
        you should consider corresponding nonlinear structures.

        6) Do not explain. Do not output JSON.

        <DATA>
        {csv_points}
        </DATA>
        """.strip()

        # 这里按 OpenAI 兼容多模态格式组织 content
        user_content = [{"type": "text", "text": text_prompt}]
        for p in image_paths:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"file://{p}"
                    },
                }
            )

        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_content),
        ]
