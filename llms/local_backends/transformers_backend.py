# sr/llm/local_backends/transformers_backend.py
from __future__ import annotations

from typing import List

import torch

from ..base import BaseLLMBackend
from ..types import LLMRequest, LLMResponse


class TransformersBackend(BaseLLMBackend):
    """
    本地 HuggingFace Transformers 后端。

    适用场景：
    - 直接加载本地模型目录
    - 直接加载 HuggingFace 模型名
    - 不起服务，直接 Python 内推理

    注意：
    - 这里默认做的是 causal LM 文本生成
    - chat 模型通常需要 tokenizer.apply_chat_template()
    - 某些模型不支持 chat template，会退化成简单字符串拼接
    """

    def __init__(self, config):
        super().__init__(config)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "需要安装 transformers：pip install transformers"
            ) from e

        model_path = config.local_model_path or config.model
        if not model_path:
            raise ValueError("TransformersBackend 需要提供 local_model_path 或 model。")

        # 解析 dtype
        torch_dtype = self._resolve_dtype(config.dtype)

        # device_map="auto" 可以让 transformers 自动分配设备
        # 如果你后面想更细地控制，也可以改成手动 to(device)
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        # 当 device=auto 时，优先让 HF 自动决定
        if config.device == "auto":
            model_kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )

        # 如果不是 auto，手动搬到指定设备
        if config.device != "auto":
            self.model = self.model.to(config.device)

        self.model.eval()

    @staticmethod
    def _resolve_dtype(dtype_str: str):
        """
        把字符串 dtype 转成 torch dtype。
        """
        if dtype_str == "float16":
            return torch.float16
        if dtype_str == "bfloat16":
            return torch.bfloat16
        if dtype_str == "float32":
            return torch.float32

        # "auto" 或其他未识别值时返回 None
        # 让 transformers 自己决定
        return None

    @staticmethod
    def _messages_to_plain_text(messages: List) -> str:
        """
        当 tokenizer 不支持 chat template 时，退化成简单文本拼接。
        """
        chunks = []
        for msg in messages:
            chunks.append(f"{msg.role}: {msg.content}")
        chunks.append("assistant: ")
        return "\n".join(chunks)

    def _build_prompt(self, request: LLMRequest) -> str:
        """
        生成输入 prompt。
        优先使用 tokenizer 的 chat template；
        如果模型没提供，则退化成手动拼接。
        """
        hf_messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # 一些 chat 模型支持 apply_chat_template
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    hf_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception:
                # 若失败，则退回纯文本拼接
                pass

        return self._messages_to_plain_text(request.messages)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        使用本地 transformers 模型做一次生成。
        """
        prompt = self._build_prompt(request)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        # 把输入张量搬到模型所在设备
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # do_sample 是否开启，通常和 temperature 有关
        do_sample = request.config.temperature > 0

        generate_kwargs = {
            "max_new_tokens": request.config.max_tokens,
            "do_sample": do_sample,
            "temperature": request.config.temperature if do_sample else None,
            "top_p": request.config.top_p if do_sample else None,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # 去掉值为 None 的字段，避免某些模型报错
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

        # 随机种子
        if request.config.seed is not None:
            torch.manual_seed(request.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request.config.seed)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        # 只取新生成部分
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]

        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return LLMResponse(
            text=text.strip(),
            raw=output_ids,
            usage={},   # 本地推理先不统计 token usage
            finish_reason="stop",
            model=request.model,
        )