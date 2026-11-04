"""Provider config wiring (no network: constructing a client makes no API call)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from trip_agents.config import gemini_api_key, get_settings
from trip_agents.llm.factory import DEFAULT_MODELS, get_chat_model


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "LLM_MODEL", "LLM_THINKING_LEVEL", "LLM_THINKING_BUDGET"):
        monkeypatch.delenv(k, raising=False)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


def test_gemini_key_falls_back_to_gemini_api_key(env: pytest.MonkeyPatch) -> None:
    env.setenv("GEMINI_API_KEY", "g-key")
    assert gemini_api_key() == "g-key"
    env.setenv("GOOGLE_API_KEY", "google-key")
    assert gemini_api_key() == "google-key"  # GOOGLE_API_KEY wins


def test_gemini_model_gets_settings(env: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langchain_google_genai")
    env.setenv("GEMINI_API_KEY", "g-key")
    env.setenv("LLM_THINKING_LEVEL", "low")
    m = get_chat_model("gemini")
    assert m.model.removeprefix("models/") == DEFAULT_MODELS["gemini"]  # type: ignore[attr-defined]
    assert m.google_api_key.get_secret_value() == "g-key"  # type: ignore[attr-defined]
    assert m.thinking_level == "low"  # type: ignore[attr-defined]
    assert m.max_retries == get_settings().llm_max_retries  # type: ignore[attr-defined]


def test_llm_model_overrides_default(env: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langchain_google_genai")
    env.setenv("GOOGLE_API_KEY", "k")
    env.setenv("LLM_MODEL", "gemini-3.5-flash-lite")
    assert get_chat_model("gemini").model.removeprefix("models/") == "gemini-3.5-flash-lite"  # type: ignore[attr-defined]
