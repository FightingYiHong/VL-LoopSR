# sr/llm/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

from .types import LLMRequest, LLMResponse


class BaseLLMBackend(ABC):
    """
    所有 LLM backend 的抽象基类。

    你的上层逻辑只需要依赖这个统一接口：
        backend.generate(request) -> LLMResponse

    这样以后无论你接：
    - OpenAI API
    - vLLM 本地服务
    - Transformers 本地模型
    - 其他 OpenAI-compatible 服务

    上层代码都不用改。
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        执行一次文本生成。
        """
        raise NotImplementedError