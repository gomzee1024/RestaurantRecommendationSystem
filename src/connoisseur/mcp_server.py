# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/mcp_server.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Connoisseur MCP server - the data + tool layer of the application.

Exposes the knowledge base, the multimodal RAG index and the multi-agent
recommender over the Model Context Protocol, so *any* MCP host (the bundled
Gradio app, Claude Desktop, an IDE, ...) can discover and use them.

Run locally over stdio:   python -m connoisseur.mcp_server
Run as a network service: connoisseur serve --transport http --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated

from fastmcp import Context, FastMCP
from pydantic import Field

from .agents.workflow import RecommendationWorkflow, format_recommendations
from .knowledge_base import KnowledgeBase
from .llm import LLMConfigurationError, get_chat_model
from .retrieval.service import RetrievalService

log = logging.getLogger("connoisseur.mcp")

mcp = FastMCP(
    "Connoisseur-Server",
    instructions=(
        "California restaurant & recipe knowledge base. Use get_restaurant_info for named places, "
        "recommend_by_vibe for atmosphere keywords, search_restaurants for open-ended or filtered "
        "requests, search_recipes / find_similar_dishes for cooking, and personalized_recommendations "
        "when the user shares their tastes, history or dietary needs."
    ),
)

kb = KnowledgeBase()
_retrieval: RetrievalService | None = None


def retrieval() -> RetrievalService:
    global _retrieval
    if _retrieval is None:
        _retrieval = RetrievalService(kb)
    return _retrieval


# =============================================================== resources
@mcp.resource("culinary-map://california", mime_type="text/plain")
def get_culinary_map() -> str:
    """The full raw California Culinary Map: free-text descriptions of 200+ restaurants
    including their vibes, cuisines, ratings and price ranges."""
    return kb.culinary_map()


@mcp.resource("knowledge-base://restaurants", mime_type="application/json")
def list_restaurants() -> str:
    """Every structured restaurant record in the knowledge base."""
    return json.dumps([r.model_dump() for r in kb.restaurants()], indent=2, ensure_ascii=False)


@mcp.resource("knowledge-base://status", mime_type="application/json")
def index_status() -> str:
    """Knowledge-base size and vector-index collection counts."""
    return json.dumps(retrieval().status(), indent=2)


# ======================================================== knowledge-base tools
@mcp.tool
def get_restaurant_info(
    restaurant_name: Annotated[
        str, Field(description="Full or partial restaurant name, e.g. 'Iron' or 'Sakura Garden'")
    ],
) -> dict:
    """Look up a restaurant by name and return its structured details: neighborhood, cuisine,
    type, rating, price range, signature dish, vibes and description."""
    matches = kb.find_restaurants(restaurant_name)
    if not matches:
        return {
            "status": "not_found",
            "message": f"No restaurant found matching '{restaurant_name}'.",
            "suggestion": "Try a partial name like 'Iron' or 'Sakura', or use search_restaurants.",
        }
    return {"status": "found", "count": len(matches), "results": [r.model_dump(exclude={"item_id"}) for r in matches]}


@mcp.tool
def recommend_by_vibe(
    vibe: Annotated[str, Field(description="One atmosphere keyword, e.g. 'moody', 'romantic', 'zen', 'sun-drenched'")],
) -> dict:
    """Find restaurants matching a vibe / atmosphere keyword. Searches structured vibe tags and
    descriptions first, then the raw culinary-map text for extra descriptive matches."""
    structured = [
        {k: getattr(r, k) for k in ("name", "neighborhood", "cuisine", "rating", "vibes", "price_range")}
        for r in kb.find_by_vibe(vibe)
    ]
    return {
        "vibe_searched": vibe,
        "structured_matches": structured,
        "raw_text_excerpts": kb.culinary_map_excerpts(vibe, limit=5),
    }


@mcp.tool
def get_review(
    restaurant_name: Annotated[str, Field(description="Full or partial restaurant name")],
) -> dict:
    """Retrieve a detailed user review of a restaurant, including the reviewer's rating,
    review text and an AI-generated description of the reviewer's photo."""
    review = kb.find_review(restaurant_name)
    if review is None:
        return {
            "status": "not_found",
            "message": f"No review found for '{restaurant_name}'. Try get_restaurant_info for its details instead.",
        }
    return {
        "status": "found",
        "restaurant": review.restaurant_name,
        "reviewer": review.reviewer,
        "rating": review.rating,
        "review_text": review.review_text,
        "image_description": review.image_description or "N/A",
        "visit_date": review.visit_date or "N/A",
    }


# ===================================================================== RAG tools
@mcp.tool
async def search_restaurants(
    query: Annotated[
        str, Field(description="Natural-language description of what the user wants, e.g. 'cozy noodles late at night'")
    ],
    neighborhood: Annotated[
        str | None, Field(description="Optional neighborhood/city filter, e.g. 'Pasadena', 'DTLA'")
    ] = None,
    cuisine: Annotated[str | None, Field(description="Optional cuisine filter, e.g. 'Italian'")] = None,
    max_price: Annotated[str | None, Field(description="Optional budget cap from '$' to '$$$$'")] = None,
    min_rating: Annotated[float | None, Field(description="Optional minimum rating out of 5", ge=0, le=5)] = None,
    k: Annotated[int, Field(description="Number of results", ge=1, le=20)] = 5,
) -> dict:
    """Semantic (vector) search over all restaurants with optional metadata filters.
    Use for open-ended requests that are not a specific restaurant name or single vibe word."""
    return await asyncio.to_thread(
        retrieval().search_restaurants, query, k, neighborhood, cuisine, max_price, min_rating
    )


@mcp.tool
async def search_recipes(
    query: Annotated[str, Field(description="What the user wants to cook, e.g. 'quick spicy vegetarian dinner'")],
    cuisine: Annotated[str | None, Field(description="Optional cuisine filter")] = None,
    k: Annotated[int, Field(ge=1, le=20)] = 5,
) -> dict:
    """Semantic search over the recipe collection (name, cuisine, ingredients and photo captions)."""
    return await asyncio.to_thread(retrieval().search_recipes, query, k, cuisine)


@mcp.tool
async def multimodal_search(
    query: Annotated[str, Field(description="Free-text food or dining query")],
    text_weight: Annotated[float, Field(ge=0, le=1, description="Weight of restaurant-text evidence")] = 0.6,
    image_weight: Annotated[float, Field(ge=0, le=1, description="Weight of dish-photo evidence")] = 0.4,
    k: Annotated[int, Field(ge=1, le=20)] = 5,
) -> dict:
    """Cross-modal search: matches the query against restaurant descriptions (MiniLM) AND dish
    photos (CLIP), normalises both score scales and returns one fused ranking."""
    return await asyncio.to_thread(retrieval().multimodal_search, query, k, text_weight, image_weight)


@mcp.tool
async def find_similar_dishes(
    recipe_name: Annotated[
        str, Field(description="Name of a recipe in the collection, e.g. 'Classic Margherita Pizza'")
    ],
    same_cuisine: Annotated[bool, Field(description="Restrict results to the same cuisine")] = False,
    k: Annotated[int, Field(ge=1, le=20)] = 5,
) -> dict:
    """Image-to-image search: find dishes whose photos look most similar to the given recipe's photo."""
    return await asyncio.to_thread(retrieval().similar_dishes, recipe_name, k, same_cuisine)


# ================================================================== agent tools
@mcp.tool
async def personalized_recommendations(
    user_context: Annotated[
        str,
        Field(
            description="Everything known about the user: tastes, past visits, posts, dietary needs, budget, occasion"
        ),
    ],
    top_n: Annotated[int, Field(ge=1, le=5)] = 3,
    ctx: Context | None = None,
) -> dict:
    """Run the six-agent recommendation workflow (profile -> retrieval -> parallel trend/style/
    nutrition analysis -> synthesis) and return grounded restaurant AND recipe picks with reasons.
    Slower than the other tools; use when the user shares personal preferences or history."""
    try:
        get_chat_model()  # fail fast with a clear message instead of six failing agents
    except LLMConfigurationError as e:
        return {"markdown": f"⚙️ **LLM not configured on the server.** {e}", "errors": [str(e)]}
    if ctx:
        await ctx.info("Running multi-agent workflow (6 agents)...")
    state = await asyncio.to_thread(RecommendationWorkflow(retrieval=retrieval()).run, user_context, top_n)
    return {
        "markdown": format_recommendations(state),
        "user_profile": state.get("user_profile", {}),
        "recommendations": state.get("final_recommendations", {}),
        "errors": state.get("errors", []),
    }


@mcp.tool
async def summarize_reviews(
    restaurant_name: Annotated[str, Field(description="Full or partial restaurant name")],
    ctx: Context,
) -> dict:
    """Summarise what diners say about a restaurant. The summary is written by the *client's*
    LLM via MCP sampling, so this server never needs model credentials for it."""
    review = kb.find_review(restaurant_name)
    if review is None:
        return {"status": "not_found", "message": f"No reviews found for '{restaurant_name}'."}
    prompt = (
        "Summarise this restaurant review in two sentences for someone deciding whether to go. "
        "Mention the standout dish and the atmosphere.\n\n"
        f"Restaurant: {review.restaurant_name}\nRating: {review.rating}\nReview: {review.review_text}\n"
        f"Photo: {review.image_description or 'n/a'}"
    )
    try:
        result = await ctx.sample(prompt, max_tokens=200)
        summary = getattr(result, "text", None) or str(result)
    except Exception as e:  # client without sampling support -> degrade gracefully
        log.info("Sampling unavailable (%s); returning raw review", e)
        summary = review.review_text
    return {"status": "found", "restaurant": review.restaurant_name, "rating": review.rating, "summary": summary}


# ======================================================================= prompts
@mcp.prompt
def plan_a_night_out(occasion: str, neighborhood: str = "anywhere in California", budget: str = "$$") -> str:
    """Reusable prompt template for planning a dinner outing."""
    return (
        f"I'm planning a {occasion} in {neighborhood} with a budget of about {budget}. "
        "Suggest two restaurants that fit, explain the vibe of each, mention a signature dish, "
        "and share what a recent reviewer thought if a review is available."
    )


# ======================================================================== health
@mcp.custom_route("/health", methods=["GET"])
async def health(_request):
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok", **retrieval().status()})


def warm_up() -> None:
    """Load the embedding models on the main thread *before* serving.

    Besides making the first query fast, this is required for stdio on Windows:
    once the transport has a thread blocked reading stdin, initialising native
    libraries (torch / scipy DLLs) from a worker thread can deadlock.
    """
    import time

    from .retrieval.vector_store import IMAGES, RESTAURANTS

    started = time.perf_counter()
    try:
        store = retrieval().store
        if store.has(RESTAURANTS):
            store.embedder.embed_texts(["warm-up"])
        if store.has(IMAGES):
            store.embedder.embed_clip_texts(["warm-up"])
        log.info("Models ready in %.1fs", time.perf_counter() - started)
    except Exception as e:  # never let warm-up take the server down
        log.warning("Warm-up skipped: %s", e)


def main(transport: str = "stdio", host: str | None = None, port: int | None = None) -> None:
    from .config import get_settings, setup_logging

    setup_logging()
    warm_up()
    if transport == "stdio":
        mcp.run(show_banner=False)  # stdout is the protocol channel; logs go to stderr
    else:
        s = get_settings()
        mcp.run(transport="http", host=host or s.mcp_host, port=port or s.mcp_port, show_banner=False)


if __name__ == "__main__":
    main()
