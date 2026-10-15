# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/llm.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Provider-agnostic LLM factory.

Every component (ingestion, agents, MCP host, sampling) gets its model from
here, so switching between IBM watsonx.ai, OpenAI and Anthropic is a single
``LLM_PROVIDER`` environment variable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings, get_settings

DEFAULT_MODELS = {
    "watsonx": {
        "chat": "ibm/granite-4-h-small",
        "vision": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    },
    "openai": {"chat": "gpt-4o-mini", "vision": "gpt-4o-mini"},
    "anthropic": {"chat": "claude-opus-5-5", "vision": "claude-opus-5-5"},
}


class LLMConfigurationError(RuntimeError):
    """Raised when the selected provider is missing credentials or packages."""


def _build(settings: Settings, model_id: str, temperature: float, max_tokens: int | None) -> BaseChatModel:
    provider = settings.llm_provider
    if provider == "watsonx":
        if not (settings.watsonx_apikey and settings.watsonx_project_id):
            raise LLMConfigurationError(
                "LLM_PROVIDER=watsonx requires WATSONX_APIKEY and WATSONX_PROJECT_ID (see .env.example)."
            )
        from langchain_ibm import ChatWatsonx

        # watsonx silently caps unspecified max_tokens at 1024, which can truncate JSON replies
        params: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens or 4096}
        return ChatWatsonx(
            model_id=model_id,
            url=settings.watsonx_url,
            apikey=settings.watsonx_apikey,
            project_id=settings.watsonx_project_id,
            params=params,
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMConfigurationError("LLM_PROVIDER=openai requires OPENAI_API_KEY (see .env.example).")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:  # pragma: no cover
            raise LLMConfigurationError("pip install langchain-openai") from e
        return ChatOpenAI(
            model=model_id, temperature=temperature, max_tokens=max_tokens, api_key=settings.openai_api_key
        )
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMConfigurationError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY (see .env.example).")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:  # pragma: no cover
            raise LLMConfigurationError("pip install langchain-anthropic") from e
        # Current Claude models reject sampling parameters (temperature) with a 400, so none is sent.
        # Thinking is always on and counts towards max_tokens, so small caps get a floor.
        return ChatAnthropic(
            model=model_id,
            max_tokens=max(max_tokens, 4096) if max_tokens else 16000,
            api_key=settings.anthropic_api_key,
        )
    raise LLMConfigurationError(f"Unknown LLM_PROVIDER '{provider}'")


def get_chat_model(temperature: float | None = None, max_tokens: int | None = None) -> BaseChatModel:
    """Chat / tool-calling model. A fresh instance per call avoids shared state between requests."""
    s = get_settings()
    model_id = s.llm_model or DEFAULT_MODELS[s.llm_provider]["chat"]
    return _build(s, model_id, s.llm_temperature if temperature is None else temperature, max_tokens)


def get_vision_model(max_tokens: int = 300) -> BaseChatModel:
    s = get_settings()
    model_id = s.vision_model or DEFAULT_MODELS[s.llm_provider]["vision"]
    return _build(s, model_id, 0.2, max_tokens)


def message_text(content: Any) -> str:
    """Flatten a LangChain message ``content`` (str or list of blocks) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content).strip()
    return str(content)


def _close_open_brackets(text: str) -> str:
    """Append the closing brackets (and quote) a model left open, e.g. ``{"a": [1, 2]`` -> ``{"a": [1, 2]}``."""
    closers: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            closers.append("}" if ch == "{" else "]")
        elif ch in "}]" and closers and closers[-1] == ch:
            closers.pop()
    return text.rstrip() + ('"' if in_string else "") + "".join(reversed(closers))


def parse_json_object(text: str) -> dict:
    """Extract the first JSON object from an LLM reply.

    Tolerates surrounding prose, ``` fences and missing final closing brackets (Granite often drops
    the outermost ``}``).
    """
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text, start)  # stops at the end of the object, ignoring trailing prose
    except json.JSONDecodeError:
        obj, _ = decoder.raw_decode(_close_open_brackets(text[start:]))
    if not isinstance(obj, dict):
        raise ValueError("Model output is not a JSON object")
    return obj
