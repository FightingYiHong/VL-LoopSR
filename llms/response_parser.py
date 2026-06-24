# llms/response_parser.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


class ResponseParser:
    """
    ResponseParser 负责把 LLM 的原始文本输出解析成结构化对象。

    设计目标：
    1. 尽量容忍模型输出中夹带解释文字
    2. 尽量从 markdown code fence / 普通文本里提取 JSON
    3. 为 proposal/meta/refiner 提供专门解析函数

    常见问题：
    - 模型会输出：
        Here is the JSON:
        ```json
        {...}
        ```
      这就需要先提取 JSON 部分。
    - 模型有时会输出合法 JSON，也有时会有尾逗号、单引号等轻微错误。
      第一版我们先只做“轻量容错”，不过度修复。
    """

    # =========================
    # 基础 JSON 提取与解析
    # =========================
    @staticmethod
    def extract_code_fence_json(text: str) -> Optional[str]:
        """
        尝试从 markdown 代码块中提取 JSON。

        支持：
        ```json
        {...}
        ```

        也支持不写 json 标记的 fenced block。
        """
        if not text:
            return None

        # 优先匹配 ```json ... ```
        pattern_json = r"```json\s*(.*?)\s*```"
        m = re.search(pattern_json, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # 再匹配普通 ``` ... ```
        pattern_plain = r"```\s*(.*?)\s*```"
        m = re.search(pattern_plain, text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()

        return None

    @staticmethod
    def extract_outermost_json(text: str) -> Optional[str]:
        """
        尝试从普通文本里提取最外层 JSON 对象或数组。

        这是一个简化版的“括号配对提取”：
        - 优先找 { ... }
        - 找不到再尝试 [ ... ]

        注意：
        这不是完美解析器，但对大多数 LLM 输出已经够用。
        """
        if not text:
            return None

        # 先尝试提取对象 {...}
        obj = ResponseParser._extract_balanced_block(text, "{", "}")
        if obj is not None:
            return obj.strip()

        # 再尝试提取数组 [...]
        arr = ResponseParser._extract_balanced_block(text, "[", "]")
        if arr is not None:
            return arr.strip()

        return None

    @staticmethod
    def _extract_balanced_block(text: str, open_char: str, close_char: str) -> Optional[str]:
        """
        从文本中抽取第一个括号平衡的块。
        例如：
        文本里有 "abc { ... } def"
        则返回 "{ ... }"
        """
        start = text.find(open_char)
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        return None

    @staticmethod
    def safe_json_loads(text: str) -> Optional[Any]:
        """
        尝试把字符串解析成 JSON。
        这里只做最轻量处理，不做过度修补。

        返回：
        - 成功：Python 对象
        - 失败：None
        """
        if not text:
            return None

        text = text.strip()

        try:
            return json.loads(text)
        except Exception:
            return None

    def parse_json_like_text(self, text: str) -> Optional[Any]:
        """
        综合解析入口：
        1. 先尝试整段文本直接 json.loads
        2. 再尝试从 code fence 提取
        3. 再尝试从普通文本里抽最外层 JSON

        返回：
        - 成功：解析出的 Python 对象
        - 失败：None
        """
        # 1) 直接解析整段文本
        obj = self.safe_json_loads(text)
        if obj is not None:
            return obj

        # 2) 从 fenced code block 中提取
        fenced = self.extract_code_fence_json(text)
        if fenced:
            obj = self.safe_json_loads(fenced)
            if obj is not None:
                return obj

        # 3) 从普通文本提取最外层 JSON
        outer = self.extract_outermost_json(text)
        if outer:
            obj = self.safe_json_loads(outer)
            if obj is not None:
                return obj

        return None

    # =========================
    # Proposal 解析
    # =========================
    def parse_proposal_response(self, text: str) -> Dict[str, Any]:
        """
        解析 proposal generator 的输出。
        优先走 JSON 解析；如果失败，则退化到正则抽取 expression。
        """
        parsed = self.parse_json_like_text(text)

        # 情况 1：标准 JSON 解析成功
        if isinstance(parsed, dict):
            candidates = parsed.get("candidates", [])
            if not isinstance(candidates, list):
                candidates = []

            normalized_candidates = []
            for item in candidates:
                if not isinstance(item, dict):
                    continue

                expression = str(item.get("expression", "")).strip()
                skeleton = str(item.get("skeleton", "")).strip()
                rationale = str(item.get("rationale", "")).strip()

                parameters = item.get("parameters", [])
                if not isinstance(parameters, list):
                    parameters = []

                prior_score = item.get("prior_score", 0.0)
                try:
                    prior_score = float(prior_score)
                except Exception:
                    prior_score = 0.0

                if expression:
                    normalized_candidates.append(
                        {
                            "expression": expression,
                            "skeleton": skeleton,
                            "parameters": [str(p) for p in parameters],
                            "rationale": rationale,
                            "prior_score": prior_score,
                        }
                    )

            if normalized_candidates:
                return {"candidates": normalized_candidates}

        # 情况 2：JSON 失败时，兜底抽 expression
        fallback_candidates = []

        import re
        expr_matches = re.findall(r'"expression"\s*:\s*"([^"]+)"', text)

        for expr in expr_matches:
            expr = expr.strip()
            if expr:
                fallback_candidates.append(
                    {
                        "expression": expr,
                        "skeleton": "",
                        "parameters": [],
                        "rationale": "fallback extracted from partial response",
                        "prior_score": 0.0,
                    }
                )

        # 去重
        seen = set()
        unique_candidates = []
        for item in fallback_candidates:
            expr = item["expression"]
            if expr not in seen:
                unique_candidates.append(item)
                seen.add(expr)

        return {"candidates": unique_candidates}

    # =========================
    # Meta 解析
    # =========================
    def parse_meta_response(self, text: str) -> Dict[str, Any]:
        """
        解析 meta llm 的输出。

        目标格式：
        {
        "refine": true,
        "reason": "...",
        "focus": [...],
        "strategy": "local_rewrite"
        }

        如果 JSON 不完整，则尽量从半截文本里兜底抽字段。
        """
        parsed = self.parse_json_like_text(text)

        # 情况 1：标准 JSON 成功
        if isinstance(parsed, dict):
            refine = parsed.get("refine", False)
            if not isinstance(refine, bool):
                if isinstance(refine, str):
                    refine = refine.strip().lower() == "true"
                else:
                    refine = False

            reason = str(parsed.get("reason", "")).strip()

            focus = parsed.get("focus", [])
            if not isinstance(focus, list):
                focus = []

            strategy = str(parsed.get("strategy", "stop")).strip()
            if not strategy:
                strategy = "stop"

            return {
                "refine": refine,
                "reason": reason,
                "focus": [str(x) for x in focus],
                "strategy": strategy,
            }

        # 情况 2：fallback，从半截文本里抽字段
        import re

        refine = False
        reason = "Fallback parsed from partial response."
        focus = []
        strategy = "stop"

        m = re.search(r'"refine"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
        if m:
            refine = m.group(1).lower() == "true"

        m = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        if m:
            reason = m.group(1).strip()

        m = re.search(r'"strategy"\s*:\s*"([^"]*)"', text)
        if m:
            strategy = m.group(1).strip() or "stop"

        return {
            "refine": refine,
            "reason": reason,
            "focus": focus,
            "strategy": strategy,
        }

    # =========================
    # Refiner 解析
    # =========================
    def parse_refiner_response(self, text: str) -> Dict[str, Any]:
        """
        解析 expression refiner 的输出。
        若 JSON 不完整，则 fallback 抽 expression。
        """
        parsed = self.parse_json_like_text(text)

        if isinstance(parsed, dict):
            refined_candidates = parsed.get("refined_candidates", [])
            if not isinstance(refined_candidates, list):
                refined_candidates = []

            normalized = []
            for item in refined_candidates:
                if not isinstance(item, dict):
                    continue

                expr = str(item.get("expression", "")).strip()
                if not expr:
                    continue

                normalized.append(
                    {
                        "expression": expr,
                        "edit_type": str(item.get("edit_type", "")).strip(),
                        "based_on": str(item.get("based_on", "")).strip(),
                        "rationale": str(item.get("rationale", "")).strip(),
                    }
                )

            if normalized:
                return {"refined_candidates": normalized}

        # fallback：从半截文本里抽 expression
        import re

        fallback = []
        expr_matches = re.findall(r'"expression"\s*:\s*"([^"]+)"', text)

        seen = set()
        for expr in expr_matches:
            expr = expr.strip()
            if expr and expr not in seen:
                fallback.append(
                    {
                        "expression": expr,
                        "edit_type": "fallback",
                        "based_on": "",
                        "rationale": "fallback extracted from partial response",
                    }
                )
                seen.add(expr)

        return {"refined_candidates": fallback}

    # =========================
    # 调试辅助函数
    # =========================
    def debug_parse(self, text: str) -> Dict[str, Any]:
        """
        给调试用：
        把解析过程中的几个关键中间结果也返回，方便看模型到底输出了什么。
        """
        fenced = self.extract_code_fence_json(text)
        outer = self.extract_outermost_json(text)
        parsed = self.parse_json_like_text(text)

        return {
            "raw_text": text,
            "fenced_json": fenced,
            "outer_json": outer,
            "parsed": parsed,
        }

    def parse_answer_tag_expression(self, text: str, target_name: str = "y") -> Dict[str, Any]:
        """
        解析 <ANSWER>y = ...</ANSWER> 这种单表达式输出。
        """
        import re

        if not text:
            return {"expression": ""}

        m = re.search(r"<ANSWER>\s*" + re.escape(target_name) + r"\s*=\s*(.*?)\s*</ANSWER>", text, flags=re.DOTALL)
        if m:
            expr = m.group(1).strip()
            return {"expression": expr}

        # fallback：如果模型没包 ANSWER 标签，但直接输出了 y = ...
        m = re.search(r"\b" + re.escape(target_name) + r"\s*=\s*(.+)", text)
        if m:
            expr = m.group(1).strip()
            return {"expression": expr}

        return {"expression": ""}