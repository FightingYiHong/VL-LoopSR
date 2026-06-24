# llms/api_backends/custom_completion_backend.py
from __future__ import annotations

from typing import Any, Dict, List

import requests

from ..base import BaseLLMBackend
from ..types import LLMRequest, LLMResponse


class CustomCompletionBackend(BaseLLMBackend):
    """
    适配你当前本地 /completions 接口的后端。

    你的服务目前看起来不是 OpenAI-compatible chat 接口，
    而是一个自定义的 prompt-completion 接口，典型请求形式类似：

    {
        "prompt": "...",
        "repeat_prompt": 1,
        "params": {
            "max_new_tokens": 256,
            "temperature": 0.2,
            "top_p": 0.95
        }
    }

    同时，你当前服务的返回格式至少出现过：
    {
        "content": ["生成文本 ..."]
    }

    所以这个 backend 做了两件事：
    1. 把 chat-style messages 转成普通 prompt
    2. 把多种可能的返回格式统一解析成 LLMResponse.text
    """

    def __init__(self, config):
        super().__init__(config)

        if not config.api_base_url:
            raise ValueError("CustomCompletionBackend 需要提供 api_base_url。")

        # 去掉末尾斜杠，避免 URL 拼接时出现双斜杠
        self.url = config.api_base_url.rstrip("/")

    @staticmethod
    def _messages_to_prompt(messages) -> str:
        """
        把 chat 风格的 messages 转成单个 prompt 字符串。

        例如：
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]

        会变成：

        [SYSTEM]
        ...

        [USER]
        ...

        [ASSISTANT]
        """
        parts: List[str] = []

        for msg in messages:
            role = msg.role.lower().strip()
            content = msg.content

            if role == "system":
                parts.append(f"[SYSTEM]\n{content}")
            elif role == "user":
                parts.append(f"[USER]\n{content}")
            elif role == "assistant":
                parts.append(f"[ASSISTANT]\n{content}")
            else:
                # 兜底处理，避免未知 role 导致崩掉
                parts.append(f"[{role.upper()}]\n{content}")

        # 追加 assistant 前缀，提示模型继续生成回答
        parts.append("[ASSISTANT]\n")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_text_from_response(data: Any) -> str:
        """
        从后端返回的 JSON 中提取最终文本。

        这个函数是“第六点”的核心：
        你的服务返回过这种结构：
            {"content": ["..."]}

        所以这里需要显式支持 content 字段。
        同时也兼容其他常见返回形式，避免以后你换服务又得重写。
        """
        text = ""

        # 只处理 dict 类型的 JSON 顶层对象
        if not isinstance(data, dict):
            return text

        # 1. 最常见：text 字段
        if "text" in data:
            value = data["text"]
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return "\n".join(str(x) for x in value)
            return str(value)

        # 2. 另一种常见命名：response
        if "response" in data:
            value = data["response"]
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return "\n".join(str(x) for x in value)
            return str(value)

        # 3. 有些服务用 generated_text
        if "generated_text" in data:
            value = data["generated_text"]
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return "\n".join(str(x) for x in value)
            return str(value)

        # 4. 你当前服务最关键的一种：content
        if "content" in data:
            value = data["content"]

            # 例如 {"content": ["abc", "def"]}
            if isinstance(value, list):
                return "\n".join(str(x) for x in value)

            # 例如 {"content": "abc"}
            if isinstance(value, str):
                return value

            # 其他情况也尽量转成字符串
            return str(value)

        # 5. OpenAI 风格 choices 兜底
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            choice0 = data["choices"][0]

            if isinstance(choice0, dict):
                # completion 风格
                if "text" in choice0:
                    return str(choice0["text"])

                # chat 风格 message.content
                if "message" in choice0 and isinstance(choice0["message"], dict):
                    msg = choice0["message"]
                    if "content" in msg:
                        return str(msg["content"])

        # 如果都没有匹配上，就返回空串
        return text

    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        执行一次自定义 completion 请求。
        """
        prompt = self._messages_to_prompt(request.messages)

        # 组装请求体
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "repeat_prompt": 1,
            "params": {
                "max_new_tokens": request.config.max_tokens,
                "temperature": request.config.temperature,
                "top_p": request.config.top_p,
                # 对自定义服务更稳一点
                "do_sample": request.config.temperature > 0,
            },
        }

        # 可选停止词
        if request.config.stop is not None:
            payload["params"]["stop"] = request.config.stop

        # 可选随机种子
        if request.config.seed is not None:
            payload["params"]["seed"] = request.config.seed

        # 允许额外参数透传
        if request.extra_body:
            payload.update(request.extra_body)

        # 关键处理：
        # 不信任系统代理环境变量，避免访问 127.0.0.1 时被 SOCKS 代理干扰
        session = requests.Session()
        session.trust_env = False

        response = session.post(
            self.url,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        data = response.json()
        text = self._extract_text_from_response(data)

        return LLMResponse(
            text=text.strip(),
            raw=data,
            usage={},
            finish_reason=None,
            model=request.model,
        )