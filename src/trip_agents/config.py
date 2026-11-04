"""Centralised runtime configuration, read from environment variables / `.env`."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Provider = Literal["mock", "openai", "anthropic", "gemini"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- model layer -------------------------------------------------------
    llm_provider: Provider = "mock"
    # Empty -> provider default from `trip_agents.llm.factory.DEFAULT_MODELS`.
    llm_model: str = ""
    llm_temperature: float = 0.2
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 6
    # Gemini "thinking" spends extra output tokens on reasoning. Gemini 3+ models take a level,
    # 2.5 models take a token budget (0 = off). None = model default.
    llm_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    llm_thinking_budget: int | None = None

    # --- tools ---------------------------------------------------------------
    tool_backend: Literal["mock", "real"] = "mock"
    # mcp: Research discovers/calls tools through an MCP stdio server; local: in-process (faster, used by evals)
    tool_transport: Literal["mcp", "local"] = "mcp"

    # --- agent behaviour -----------------------------------------------------
    max_revisions: int = Field(default=2, ge=0, description="Itinerary re-plan rounds after rule failures")
    structured_max_retries: int = Field(default=3, ge=1)
    research_max_steps: int = Field(default=8, ge=1)
    research_max_rounds_per_tool: int = Field(default=3, ge=1)

    # --- semantic cache (off by default: a hit skips research and candidate picking) ---
    semantic_cache: bool = False
    cache_threshold: float = 0.85
    cache_ttl_days: float = 7

    # --- persistence ---------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / ".data"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def cache_db(self) -> Path:
        return self.data_dir / "semantic_cache.sqlite"

    @property
    def store_db(self) -> Path:
        return self.data_dir / "store.sqlite"


def gemini_api_key() -> str | None:
    """GOOGLE_API_KEY, falling back to GEMINI_API_KEY (both names are used in Google's docs)."""
    get_settings()  # make sure .env has been exported
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or None


def anthropic_api_key() -> str | None:
    """ANTHROPIC_API_KEY, falling back to CLAUDE_REWRITER_API_KEY (the name used for the rewriter's key)."""
    get_settings()
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_REWRITER_API_KEY") or None


@lru_cache
def get_settings() -> Settings:
    # Provider SDKs read keys from os.environ, so export .env there too (real env vars win).
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings()
