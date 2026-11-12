"""Claude (Anthropic API) backend.

Credentials come from the environment, exactly as the official SDK resolves
them: ``ANTHROPIC_API_KEY`` (or ``ANTHROPIC_AUTH_TOKEN`` / an ``ant auth login``
profile). Never hard-code a key in a config file.

Notes on the request shape:
* Current Claude models do not accept ``temperature``; every call is sampled with
  the model's default settings, so repeated calls already give the varied answers
  the consistency features need.
* ``output_config.effort`` controls how hard the model thinks. The hidden
  reasoning is not returned, but it is counted in ``usage.output_tokens``, which
  we keep as a feature.
* For models that support it we enable server-side refusal fallbacks, so a
  declined request is re-run on a fallback model inside the same call.
"""

from __future__ import annotations

from typing import Any

from .base import LLMClient, LLMResponse

# Models that accept the server-side ``fallbacks: "default"`` parameter.
FALLBACK_MODELS = {
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        max_tokens: int = 16000,
        effort: str | None = "medium",
        fallbacks: str | bool = "auto",
        max_retries: int = 5,
        timeout_seconds: float = 600.0,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "The 'anthropic' package is required for provider: anthropic. "
                    "Install it with `pip install anthropic`."
                ) from exc
            client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout_seconds)
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        if fallbacks == "auto":
            self.use_fallbacks = model in FALLBACK_MODELS
        else:
            self.use_fallbacks = bool(fallbacks)

    def build_request(self, prompt: str) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        if self.use_fallbacks:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    def complete(self, prompt, *, purpose="answer", meta=None) -> LLMResponse:
        request = self.build_request(prompt)
        if self.use_fallbacks:
            message = self.client.beta.messages.create(**request)
        else:
            message = self.client.messages.create(**request)

        # Always check stop_reason before reading content: a refusal has no answer.
        text = ""
        if message.stop_reason != "refusal":
            text = "".join(
                getattr(block, "text", "")
                for block in message.content
                if getattr(block, "type", None) == "text"
            )
        usage = getattr(message, "usage", None)
        return LLMResponse(
            text=text,
            model=getattr(message, "model", self.model),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            stop_reason=message.stop_reason,
            extra={"request_id": getattr(message, "_request_id", None)},
        )
