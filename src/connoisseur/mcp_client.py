# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/mcp_client.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Diagnostic MCP client: verifies the server end-to-end over the real protocol.

connoisseur check            # discovery + one call per tool (no LLM needed)
connoisseur check --sampling # also exercise the sampling round-trip (needs an LLM key)
"""

from __future__ import annotations

import json

from .mcp_connection import make_client, project_roots

REQUIRED_TOOLS = {
    "get_restaurant_info",
    "recommend_by_vibe",
    "get_review",
    "search_restaurants",
    "search_recipes",
    "multimodal_search",
    "find_similar_dishes",
    "personalized_recommendations",
    "summarize_reviews",
}


def _payload(result) -> dict:
    if getattr(result, "structured_content", None):
        return result.structured_content
    return json.loads(result.content[0].text)


def _short(obj, n: int = 400) -> str:
    s = json.dumps(obj, indent=2, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "\n  ..."


async def run_checks(with_sampling: bool = False) -> bool:
    ok = True
    async with make_client(with_sampling=with_sampling, keep_alive=False) as client:
        print("=" * 64 + "\nMCP connection verification\n" + "=" * 64)

        tools = await client.list_tools()
        names = {t.name for t in tools}
        print(f"\nDiscovered {len(tools)} tools:")
        for t in tools:
            print(f"  - {t.name}: {(t.description or '').splitlines()[0][:80]}")
        missing = REQUIRED_TOOLS - names
        if missing:
            ok = False
            print(f"  ✗ MISSING: {sorted(missing)}")
        else:
            print("  ✓ all required tools present")

        resources = await client.list_resources()
        print(f"\nDiscovered {len(resources)} resources:")
        for r in resources:
            print(f"  - {r.uri}: {r.name}")

        prompts = await client.list_prompts()
        print(f"\nDiscovered {len(prompts)} prompts:")
        for p in prompts:
            print(f"  - {p.name}")

        roots = project_roots()
        print(f"\nConfigured {len(roots)} root(s):")
        for root in roots:
            print(f"  - {root.name}: {root.uri}")

        status = json.loads((await client.read_resource("knowledge-base://status"))[0].text)
        print(f"\nKnowledge base status: {status}")

        demos = [
            ("get_restaurant_info", {"restaurant_name": "Iron & Embers"}),
            ("recommend_by_vibe", {"vibe": "moody"}),
            ("get_review", {"restaurant_name": "Iron & Embers"}),
            ("search_restaurants", {"query": "romantic Italian dinner", "max_price": "$$$", "k": 3}),
            ("search_recipes", {"query": "spicy noodles", "k": 3}),
            ("multimodal_search", {"query": "fresh sushi with minimalist presentation", "k": 4}),
            ("find_similar_dishes", {"recipe_name": "Classic Margherita Pizza", "k": 3}),
        ]
        if with_sampling:
            demos.append(("summarize_reviews", {"restaurant_name": "Sakura Garden"}))
        for name, args in demos:
            print("\n" + "-" * 64 + f"\n{name}({args})\n" + "-" * 64)
            try:
                print(_short(_payload(await client.call_tool(name, args))))
            except Exception as e:
                ok = False
                print(f"  ✗ {type(e).__name__}: {e}")

    print("\n" + ("✅ All checks passed" if ok else "❌ Some checks failed"))
    return ok
