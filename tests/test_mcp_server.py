# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : tests/test_mcp_server.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Exercises the MCP server over the real protocol using fastmcp's in-memory transport."""

import json

from fastmcp import Client

from connoisseur.mcp_client import REQUIRED_TOOLS
from connoisseur.mcp_server import mcp


def payload(result) -> dict:
    return result.structured_content or json.loads(result.content[0].text)


async def test_discovery():
    async with Client(mcp) as client:
        assert {t.name for t in await client.list_tools()} >= REQUIRED_TOOLS
        uris = {str(r.uri) for r in await client.list_resources()}
        assert {"culinary-map://california", "knowledge-base://restaurants", "knowledge-base://status"} <= uris
        assert "plan_a_night_out" in {p.name for p in await client.list_prompts()}


async def test_lab_tools():
    async with Client(mcp) as client:
        info = payload(await client.call_tool("get_restaurant_info", {"restaurant_name": "Iron"}))
        assert info["status"] == "found" and info["results"][0]["signature_dish"] == "45-day dry-aged ribeye"

        missing = payload(await client.call_tool("get_restaurant_info", {"restaurant_name": "Nope"}))
        assert missing["status"] == "not_found"

        vibe = payload(await client.call_tool("recommend_by_vibe", {"vibe": "zen"}))
        assert [m["name"] for m in vibe["structured_matches"]] == ["Sakura Garden"]
        assert vibe["raw_text_excerpts"]

        review = payload(await client.call_tool("get_review", {"restaurant_name": "Iron & Embers"}))
        assert review["reviewer"] == "SteakFan"
        assert payload(await client.call_tool("get_review", {"restaurant_name": "Sakura"}))["status"] == "not_found"


async def test_rag_tool_and_resource():
    async with Client(mcp) as client:
        res = payload(await client.call_tool("search_restaurants", {"query": "casual seafood tacos", "max_price": "$"}))
        assert [r["name"] for r in res["results"]] == ["Mar de Cortez"]

        kb = json.loads((await client.read_resource("knowledge-base://restaurants"))[0].text)
        assert len(kb) == 3


async def test_summarize_reviews_uses_client_sampling():
    seen = {}

    async def sampler(messages, params, context):
        seen["prompt"] = messages[0].content.text
        return "Great steak, moody room."

    async with Client(mcp, sampling_handler=sampler) as client:
        out = payload(await client.call_tool("summarize_reviews", {"restaurant_name": "Iron"}))
    assert out["summary"] == "Great steak, moody room."
    assert "ribeye" in seen["prompt"]
