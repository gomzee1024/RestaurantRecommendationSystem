# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/config.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Centralised runtime configuration.

All settings come from environment variables (or a local ``.env`` file), so no
credentials are ever hard-coded. See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Source checkout -> the repo root; regular (non-editable) install -> the working directory.
_SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _SRC_ROOT if (_SRC_ROOT / "pyproject.toml").exists() else Path.cwd()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM provider -------------------------------------------------------
    llm_provider: Literal["watsonx", "openai", "anthropic"] = "watsonx"
    llm_model: str | None = None  # chat / tool-calling model; provider default if unset
    vision_model: str | None = None  # image captioning model; provider default if unset
    llm_temperature: float = 0.7

    # IBM watsonx.ai
    watsonx_apikey: str | None = None
    watsonx_url: str = "https://us-south.ml.cloud.ibm.com"
    watsonx_project_id: str | None = None

    # OpenAI / Anthropic (keys are read by their SDKs: OPENAI_API_KEY / ANTHROPIC_API_KEY)

    # ---- Data & index -------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model: str = "openai/clip-vit-base-patch32"

    # ---- MCP ----------------------------------------------------------------
    # When set, the host connects to a remote MCP server over HTTP instead of
    # spawning a local stdio subprocess (used by docker-compose).
    mcp_server_url: str | None = None
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000

    # ---- Web UI -------------------------------------------------------------
    ui_host: str = "127.0.0.1"
    ui_port: int = 7860

    # ---- Derived paths ------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def culinary_map_path(self) -> Path:
        return self.raw_dir / "California-Culinary-Map.txt"

    @property
    def restaurants_path(self) -> Path:
        return self.data_dir / "restaurants.json"

    @property
    def reviews_path(self) -> Path:
        return self.data_dir / "reviews.json"

    @property
    def recipes_path(self) -> Path:
        return self.data_dir / "recipes.json"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "recipe_images"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def setup_logging(verbose: bool = False) -> None:
    """Log to stderr (stdout is the MCP stdio channel) and silence chatty HTTP clients."""
    import logging
    import sys

    # Configure our own logger explicitly: libraries such as fastmcp install root
    # handlers first, which would turn a plain basicConfig() into a silent no-op.
    app_logger = logging.getLogger("connoisseur")
    if not app_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        app_logger.addHandler(handler)
    app_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    app_logger.propagate = False
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "chromadb.telemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
