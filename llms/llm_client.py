# sr/llm/llm_client.py
from __future__ import annotations

from typing import Dict, List, Optional
from llms.types import Message
from .registry import get_backend
from .types import BackendConfig, GenerationConfig, LLMRequest, LLMResponse, Message


class LLMClient:
    """
    对上层暴露的统一 LLM 客户端。

    上层不需要知道底层到底是：
    - API 调用
    - 本地 transformers
    - 其他 backend

    只需要：
        client.generate(...)
    """

    def __init__(self, backend_config: BackendConfig):
        self.backend_config = backend_config
        self.backend = get_backend(backend_config)

    def generate(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        extra_body: Optional[Dict] = None,
    ) -> LLMResponse:
        """
        最常用的统一调用接口。

        参数说明：
        - messages: chat 消息列表
        - model: 若不传，则默认使用 backend_config.model
        - temperature / max_tokens / top_p: 生成参数
        - stop: 停止词
        - seed: 随机种子
        - extra_body: 透传给特定 backend 的附加参数
        """
        req = LLMRequest(
            messages=messages,
            model=model or self.backend_config.model,
            config=GenerationConfig(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                seed=seed,
            ),
            extra_body=extra_body or {},
        )

        return self.backend.generate(req)

    def generate_from_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        extra_body: Optional[Dict] = None,
    ) -> LLMResponse:
        """
        一个更方便的辅助接口。
        适合你现在快速测试 proposal / refine 这类模块。
        """
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        return self.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            seed=seed,
            extra_body=extra_body,
        )