# sr/llm/api_backends/openai_compatible_backend.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from ..base import BaseLLMBackend
from ..types import LLMRequest, LLMResponse


class OpenAICompatibleBackend(BaseLLMBackend):
    """
    OpenAI-compatible 后端。

    适用场景：
    - OpenAI 官方 API
    - vLLM OpenAI-compatible server
    - 本地部署的兼容 /v1/chat/completions 的服务
    - LM Studio / Ollama 的兼容接口（如果已开启兼容模式）

    使用方式：
        config.backend_type = "openai_compatible"
        config.api_base_url = "http://127.0.0.1:8000/v1"
        config.api_key = "EMPTY"
    """

    def __init__(self, config):
        super().__init__(config)

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "需要安装 openai 包：pip install openai"
            ) from e

        if not config.api_base_url:
            raise ValueError("OpenAICompatibleBackend 需要提供 api_base_url。")

        # 初始化 OpenAI 风格客户端。对本地 vLLM，长时间自动重试会把一次
        # VLM timeout 放大成多倍耗时；需要时可用环境变量显式控制。
        client_kwargs = {
            "base_url": config.api_base_url,
            "api_key": config.api_key or "EMPTY",
            "timeout": config.timeout,
        }
        max_retries = os.environ.get("LLMSR_OPENAI_MAX_RETRIES")
        if max_retries is not None:
            client_kwargs["max_retries"] = int(max_retries)
        self.client = OpenAI(**client_kwargs)

    @staticmethod
    def _convert_messages(messages) -> List[Dict[str, str]]:
        """
        把我们自己定义的 Message 列表转成 OpenAI chat 格式。
        """
        converted = []
        for msg in messages:
            converted.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )
        return converted

    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        发起一次 chat completion 请求，并统一返回成 LLMResponse。
        """
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": self._convert_messages(request.messages),
            "temperature": request.config.temperature,
            "max_tokens": request.config.max_tokens,
            "top_p": request.config.top_p,
        }

        # 可选停止词
        if request.config.stop is not None:
            payload["stop"] = request.config.stop

        # 可选随机种子（有些兼容服务支持）
        if request.config.seed is not None:
            payload["seed"] = request.config.seed

        # 额外参数透传
        if request.extra_body:
            payload.update(request.extra_body)

        completion = self.client.chat.completions.create(**payload)

        text = ""
        finish_reason = None
        model_name = getattr(completion, "model", request.model)
        usage = {}

        # 取第一条候选
        if getattr(completion, "choices", None):
            choice = completion.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)

            # 标准 chat completion 返回里，内容通常在 choice.message.content
            if getattr(choice, "message", None) is not None:
                text = getattr(choice.message, "content", "") or ""

        # token 使用量
        if getattr(completion, "usage", None) is not None:
            usage_obj = completion.usage
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }

        return LLMResponse(
            text=text,
            raw=completion,
            usage=usage,
            finish_reason=finish_reason,
            model=model_name,
        )
