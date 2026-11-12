"""Backend for any server that speaks the OpenAI chat-completions API.

This covers hosted APIs and - more usefully for a student budget - free local
models served by Ollama (``http://localhost:11434/v1``), LM Studio or vLLM.
Small local models make *more* mistakes, which gives the classifier more wrong
answers to learn from.
"""

from __future__ import annotations

import os
from typing import Any

from .base import LLMClient, LLMResponse


class OpenAICompatibleClient(LLMClient):
    provider = "openai_compatible"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_tokens: int = 2048,
        answer_temperature: float | None = 0.0,
        sample_temperature: float | None = 0.8,
        max_retries: int = 5,
        timeout_seconds: float = 600.0,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "The 'openai' package is required for provider: openai_compatible. "
                    "Install it with `pip install openai`."
                ) from exc
            # Local servers (Ollama etc.) ignore the key but the SDK requires one.
            api_key = os.environ.get(api_key_env) or "not-needed"
            client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                max_retries=max_retries,
                timeout=timeout_seconds,
            )
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.answer_temperature = answer_temperature
        self.sample_temperature = sample_temperature

    def build_request(self, prompt: str, purpose: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        temperature = self.sample_temperature if purpose == "sample" else self.answer_temperature
        if temperature is not None:
            request["temperature"] = temperature
        return request

    def complete(self, prompt, *, purpose="answer", meta=None) -> LLMResponse:
        completion = self.client.chat.completions.create(**self.build_request(prompt, purpose))
        choice = completion.choices[0]
        usage = getattr(completion, "usage", None)
        return LLMResponse(
            text=choice.message.content or "",
            model=getattr(completion, "model", self.model),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            stop_reason=getattr(choice, "finish_reason", None),
        )
