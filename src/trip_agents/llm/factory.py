"""One function to get a chat model, whatever the provider.

All agents depend only on LangChain's `BaseChatModel` interface (`invoke`,
`bind_tools`, `with_structured_output`), so switching provider is an env-var
change and the tracer callbacks work the same for every provider.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from trip_agents.config import Provider, anthropic_api_key, gemini_api_key, get_settings

# Cheap / small default per provider. Override with LLM_MODEL; check each
# provider's current model list, names change over time.
DEFAULT_MODELS: dict[str, str] = {
    "mock": "mock-planner-v1",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-5-5",
    "gemini": "gemini-3.8-flash",  # 2.5-flash returns 404 for new keys since late 2026
}


# OpenAI models that still accept a non-default temperature (checked 2026-11 against the
# live API). Newer reasoning models (gpt-5, gpt-5.5, gpt-5.6-*, gpt-6*, o-series) return
# 400 "Only the default (1) value is supported" for temperature=0, so for them we omit
# the parameter and the model runs at its default sampling.
_OPENAI_TEMPERATURE_OK = re.compile(r"^(gpt-3\.5|gpt-4|gpt-5\.4-(mini|nano))")


# Claude models that reject temperature != 1 (langchain-anthropic raises client-side).
_ANTHROPIC_NO_TEMPERATURE = ("claude-fable-5", "claude-sonnet-5-5")


def accepts_temperature(provider: str, model: str) -> bool:
    """False when the model only runs at its default temperature; callers' value is then dropped."""
    if provider == "openai":
        return bool(_OPENAI_TEMPERATURE_OK.match(model))
    if provider == "anthropic":
        return not model.startswith(_ANTHROPIC_NO_TEMPERATURE)
    return True


def get_chat_model(
    provider: Provider | None = None,
    model: str | None = None,
    *,
    temperature: float | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    s = get_settings()
    provider = provider or s.llm_provider
    model = model or s.llm_model or DEFAULT_MODELS[provider]
    temperature = s.llm_temperature if temperature is None else temperature
    if not accepts_temperature(provider, model):
        temperature = None  # omitted from the request -> provider default (1)

    if provider == "mock":
        from trip_agents.llm.mock import MockChatModel

        return MockChatModel(model_name=model, **kwargs)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, timeout=s.llm_timeout_s, **kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if (key := anthropic_api_key()) is not None:
            kwargs.setdefault("api_key", key)
        return ChatAnthropic(model_name=model, temperature=temperature, timeout=s.llm_timeout_s, **kwargs)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs.setdefault("max_retries", s.llm_max_retries)  # free tier rate-limits aggressively; SDK backs off
        if s.llm_thinking_level is not None:
            kwargs.setdefault("thinking_level", s.llm_thinking_level)
        if s.llm_thinking_budget is not None:
            kwargs.setdefault("thinking_budget", s.llm_thinking_budget)
        return ChatGoogleGenerativeAI(
            model=model, temperature=temperature, timeout=s.llm_timeout_s, google_api_key=gemini_api_key(), **kwargs
        )
    raise ValueError(f"unknown provider {provider!r}")


def model_name_of(model: BaseChatModel) -> str:
    for attr in ("model_name", "model"):
        v = getattr(model, attr, None)
        if isinstance(v, str):
            return v
    return type(model).__name__
