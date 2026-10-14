# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/host/agent.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""MCP host: a ReAct (reason + act) agent over the Connoisseur MCP server.

Tools are discovered from the server at runtime and bound to the LLM, so any
tool added to the server becomes available here without code changes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from fastmcp import Client
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..llm import get_chat_model, message_text
from ..mcp_connection import make_client

log = logging.getLogger(__name__)

MAX_STEPS = 8
MAX_TOOL_OUTPUT_CHARS = 12_000

SYSTEM_PROMPT = """You are Connoisseur Companion, a warm and knowledgeable AI guide to California's restaurant scene.
You have access to a database of California restaurants, user reviews and a recipe collection through tools:

- get_restaurant_info: look up specific restaurants by name.
- recommend_by_vibe: find restaurants matching a single mood or atmosphere word (moody, romantic, zen...).
- get_review / summarize_reviews: retrieve or summarise detailed reviews of a restaurant.
- search_restaurants: semantic search for open-ended requests, with optional neighborhood, cuisine, budget and rating filters.
- search_recipes / find_similar_dishes: recipes to cook at home and visually similar dishes.
- multimodal_search: combine restaurant descriptions and dish photos for food-centric queries.
- personalized_recommendations: run the full multi-agent pipeline when the user shares their tastes,
  history or dietary needs and wants tailored restaurant AND recipe picks.

Rules:
- Always ground answers in tool results; never invent restaurants, ratings, prices or dishes.
- If a tool returns nothing useful, say so and suggest an alternative search.
- Prefer one or two well-chosen tool calls over many.
- Answer concisely in Markdown. Use a small table when comparing several restaurants.
- If personalized_recommendations returns a "markdown" field, present it to the user (you may lightly edit it)."""


@dataclass
class AgentEvent:
    kind: str  # "tool_call" | "tool_result" | "final" | "error"
    content: str
    tool: str | None = None


def _history_to_messages(history: list[dict]) -> list:
    msgs = []
    for m in history or []:
        if (m.get("metadata") or {}).get("title"):  # skip tool-trace bubbles
            continue
        text = message_text(m.get("content", ""))
        if not text:
            continue
        msgs.append(HumanMessage(content=text) if m.get("role") == "user" else AIMessage(content=text))
    return msgs


def _tool_output(result) -> str:
    if not result.content:
        return "(no result)"
    text = " ".join(item.text if hasattr(item, "text") else str(item) for item in result.content)
    return text[:MAX_TOOL_OUTPUT_CHARS]


class ConnoisseurAgent:
    def __init__(self, client: Client | None = None):
        self.client = client or make_client()

    async def stream(self, user_message: str, history: list[dict] | None = None) -> AsyncIterator[AgentEvent]:
        async with self.client:
            mcp_tools = await self.client.list_tools()
            # MCP tool schemas -> OpenAI-style function definitions understood by bind_tools
            tool_defs = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema},
                }
                for t in mcp_tools
            ]
            model = get_chat_model().bind_tools(tool_defs)

            messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT), *_history_to_messages(history or [])]
            messages.append(HumanMessage(content=user_message))

            for _ in range(MAX_STEPS):
                response = await model.ainvoke(messages)
                messages.append(response)
                if not response.tool_calls:
                    yield AgentEvent("final", message_text(response.content))
                    return
                for call in response.tool_calls:
                    yield AgentEvent("tool_call", json.dumps(call["args"], ensure_ascii=False), call["name"])
                    try:
                        result = await self.client.call_tool(call["name"], call["args"], raise_on_error=False)
                        output = _tool_output(result)
                    except Exception as e:  # surface tool failures to the model instead of crashing
                        output = json.dumps({"status": "error", "message": str(e)})
                    yield AgentEvent("tool_result", output[:500], call["name"])
                    messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

            yield AgentEvent(
                "final", "I wasn't able to complete that request within my step budget. Please try rephrasing."
            )

    async def ask(self, user_message: str, history: list[dict] | None = None) -> str:
        final = ""
        async for event in self.stream(user_message, history):
            if event.kind == "final":
                final = event.content
        return final
