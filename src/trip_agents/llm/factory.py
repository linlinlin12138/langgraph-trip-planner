"""One function to get a chat model, whatever the provider.

All agents depend only on LangChain's `BaseChatModel` interface (`invoke`,
`bind_tools`, `with_structured_output`), so switching provider is an env-var
change and the tracer callbacks work the same for every provider.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from trip_agents.config import Provider, get_settings

# Cheap / small default per provider. Override with LLM_MODEL; check each
# provider's current model list, names change over time.
DEFAULT_MODELS: dict[str, str] = {
    "mock": "mock-planner-v1",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-5-5",
    "gemini": "gemini-2.5-flash",
}


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

    if provider == "mock":
        from trip_agents.llm.mock import MockChatModel

        return MockChatModel(model_name=model, **kwargs)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, timeout=s.llm_timeout_s, **kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model_name=model, temperature=temperature, timeout=s.llm_timeout_s, **kwargs)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature, timeout=s.llm_timeout_s, **kwargs)
    raise ValueError(f"unknown provider {provider!r}")


def model_name_of(model: BaseChatModel) -> str:
    for attr in ("model_name", "model"):
        v = getattr(model, attr, None)
        if isinstance(v, str):
            return v
    return type(model).__name__
