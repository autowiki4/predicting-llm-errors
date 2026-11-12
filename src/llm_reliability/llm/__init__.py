"""LLM backends and a factory that builds one from the config."""

from __future__ import annotations

from ..config import LLMConfig
from ..data import Question
from .base import LLMClient, LLMResponse

__all__ = ["LLMClient", "LLMResponse", "make_client"]


def make_client(cfg: LLMConfig, questions: list[Question] | None = None, seed: int = 0) -> LLMClient:
    provider = cfg.provider.lower()
    if provider == "simulated":
        from .simulated import SimulatedLLM

        if questions is None:
            raise ValueError("The simulated provider needs the list of questions")
        return SimulatedLLM(
            questions, seed=seed, difficulty_shift=cfg.simulated_difficulty_shift, model=cfg.model
        )
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            effort=cfg.effort,
            fallbacks=cfg.fallbacks,
            max_retries=cfg.max_retries,
            timeout_seconds=cfg.timeout_seconds,
        )
    if provider in {"openai_compatible", "openai", "ollama"}:
        from .openai_compat import OpenAICompatibleClient

        return OpenAICompatibleClient(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env,
            max_tokens=cfg.max_tokens,
            answer_temperature=cfg.answer_temperature,
            sample_temperature=cfg.sample_temperature,
            max_retries=cfg.max_retries,
            timeout_seconds=cfg.timeout_seconds,
        )
    raise ValueError(
        f"Unknown provider {cfg.provider!r}; use 'simulated', 'anthropic' or 'openai_compatible'"
    )
