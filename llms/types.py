# sr/llm/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """
    单条对话消息。
    统一成 chat-style 格式，便于 API 模型和本地 chat 模型共用。

    role:
        - "system"
        - "user"
        - "assistant"
    content:
        文本内容
    """
    role: str
    content: str


@dataclass
class GenerationConfig:
    """
    生成参数。
    尽量抽成统一结构，避免不同 backend 的参数散落在代码里。
    """
    temperature: float = 0.2
    max_tokens: int = 512
    top_p: float = 0.95
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


@dataclass
class LLMRequest:
    """
    统一的 LLM 请求对象。
    所有 backend 都接收这个对象，而不是各写各的参数格式。
    """
    messages: List[Message]
    model: str
    config: GenerationConfig = field(default_factory=GenerationConfig)

    # 额外参数，给某些后端留扩展口
    # 比如某些 OpenAI-compatible server 支持 extra_body 字段
    extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """
    统一的 LLM 返回对象。
    上层只关心 text 和一些基础元信息，不需要知道底层 API 长什么样。
    """
    text: str

    # 原始返回，方便调试
    raw: Any = None

    # token 使用量等信息
    usage: Dict[str, Any] = field(default_factory=dict)

    # 停止原因，例如 "stop" / "length"
    finish_reason: Optional[str] = None

    # 实际使用的模型名
    model: Optional[str] = None


@dataclass
class BackendConfig:
    """
    backend 配置。
    一个配置对象同时兼容：
    - OpenAI-compatible API
    - 本地 transformers 模型

    backend_type:
        - "openai_compatible"
        - "transformers"
    """
    backend_type: str
    model: str

    # ========== API 相关 ==========
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 120

    # ========== 本地模型相关 ==========
    local_model_path: Optional[str] = None
    device: str = "auto"   # "auto" / "cuda" / "cpu"
    dtype: str = "auto"    # "auto" / "float16" / "bfloat16" / "float32"

    # 额外自定义配置
    extra: Dict[str, Any] = field(default_factory=dict)