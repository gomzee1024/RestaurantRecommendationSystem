# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/agents/workflow.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Hybrid sequential / parallel multi-agent recommendation workflow (LangGraph).

    profile ─► retrieve ─┬─► trends ────┐
                         ├─► styles ────┼─► recommend ─► END
                         └─► nutrition ─┘

1. **User Profile Generator** turns free text (history, posts, wishes) into a profile.
2. **RAG Retriever** plans queries + filters, then runs them against the real index.
3. **Trend / Style / Nutrition** experts analyse the candidates *in parallel*.
4. **Recommendation Expert** synthesises everything, constrained to retrieved
   candidates; anything it invents is dropped (grounding guardrail).
"""

from __future__ import annotations

import json
import logging
import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ..llm import get_chat_model, message_text, parse_json_object
from ..retrieval.service import RetrievalService
from .personas import (
    NUTRITION_EXPERT,
    RAG_RETRIEVER,
    RECOMMENDER,
    STYLE_EXPERT,
    TREND_ANALYST,
    USER_PROFILER,
    AgentPersona,
)

log = logging.getLogger(__name__)


class RecommendationState(TypedDict, total=False):
    user_input: str
    top_n: int
    user_profile: dict
    search_plan: dict
    retrieved_restaurants: list[dict]
    retrieved_recipes: list[dict]
    trend_analysis: dict
    style_analysis: dict
    nutrition_analysis: dict
    final_recommendations: dict
    workflow_step: str
    errors: Annotated[list[str], operator.add]  # reducer: safe for parallel writers


def _compact(items: list[dict], keys: tuple[str, ...]) -> str:
    return json.dumps([{k: i.get(k) for k in keys if i.get(k) is not None} for i in items], ensure_ascii=False)


REST_KEYS = (
    "name",
    "neighborhood",
    "cuisine",
    "type",
    "rating",
    "price_range",
    "signature_dish",
    "vibes",
    "description",
)
RECIPE_KEYS = ("name", "cuisine", "total_time", "ingredients")


class RecommendationWorkflow:
    def __init__(
        self,
        llm_factory: Callable[[], BaseChatModel] = get_chat_model,
        retrieval: RetrievalService | None = None,
    ):
        self.llm_factory = llm_factory
        self.retrieval = retrieval or RetrievalService()
        self.graph = self._build()

    # ----------------------------------------------------------------- helper
    def _ask_json(self, persona: AgentPersona, prompt: str) -> dict:
        llm = self.llm_factory()
        messages = [SystemMessage(content=persona.system_prompt()), HumanMessage(content=prompt)]
        text = message_text(llm.invoke(messages).content)
        try:
            return parse_json_object(text)
        except ValueError as e:  # one repair round: show the model its own reply and the parse error
            log.info("%s returned invalid JSON (%s); asking it to repair", persona.key, e)
            messages += [
                AIMessage(content=text),
                HumanMessage(content=f"That was not valid JSON ({e}). Reply with only the corrected JSON object."),
            ]
            return parse_json_object(message_text(llm.invoke(messages).content))

    # ------------------------------------------------------------------ nodes
    def node_profile(self, state: RecommendationState) -> dict:
        prompt = f"""Analyze this user data and create a comprehensive dining profile:

{state["user_input"]}

Return JSON with keys:
favorite_cuisines (list), dietary_restrictions (list), dining_occasions (list),
price_range (one of "$","$$","$$$","$$$$" or "any"), preferred_neighborhoods (list),
adventurousness_score (1-10), flavor_preferences (list), vibes (list of atmosphere words),
summary (string, 1-2 sentences)."""
        try:
            profile = self._ask_json(USER_PROFILER, prompt)
            return {"user_profile": profile, "workflow_step": "profile_generated"}
        except Exception as e:
            log.warning("profile node failed: %s", e)
            return {
                "user_profile": {"summary": state["user_input"]},
                "workflow_step": "profile_generated",
                "errors": [f"profile: {e}"],
            }

    def node_retrieve(self, state: RecommendationState) -> dict:
        profile = state.get("user_profile", {})
        prompt = f"""User profile:
{json.dumps(profile, indent=2, ensure_ascii=False)}

Plan searches over a vector database of California restaurants and a recipe collection.
Return JSON:
{{"restaurant_queries": [2-3 short natural-language queries mixing cuisine, vibe and occasion],
  "recipe_queries": [1-2 short queries for dishes the user could cook at home],
  "max_price": "$".."$$$$" or null, "neighborhood": string or null}}"""
        errors: list[str] = []
        try:
            plan = self._ask_json(RAG_RETRIEVER, prompt)
        except Exception as e:
            errors.append(f"retrieval plan: {e}")
            basis = (
                " ".join(
                    profile.get("favorite_cuisines", [])
                    + profile.get("flavor_preferences", [])
                    + profile.get("vibes", [])
                )
                or state["user_input"]
            )
            plan = {"restaurant_queries": [basis], "recipe_queries": [basis]}

        max_price = plan.get("max_price") or None
        neighborhood = plan.get("neighborhood") or None
        restaurants: dict[str, dict] = {}
        for q in plan.get("restaurant_queries", [])[:3] or [state["user_input"]]:
            res = self.retrieval.search_restaurants(q, k=5, neighborhood=neighborhood, max_price=max_price)
            if not res["results"] and (neighborhood or max_price):  # filters too strict -> relax
                res = self.retrieval.search_restaurants(q, k=5)
            for r in res["results"]:
                restaurants.setdefault(r["name"], r)
        recipes: dict[str, dict] = {}
        for q in plan.get("recipe_queries", [])[:2] or [state["user_input"]]:
            for r in self.retrieval.search_recipes(q, k=5)["results"]:
                recipes.setdefault(r["name"], r)

        return {
            "search_plan": plan,
            "retrieved_restaurants": list(restaurants.values())[:8],
            "retrieved_recipes": list(recipes.values())[:8],
            "workflow_step": "candidates_retrieved",
            "errors": errors,
        }

    def _analysis_context(self, state: RecommendationState) -> str:
        return (
            f"User profile: {json.dumps(state.get('user_profile', {}), ensure_ascii=False)}\n"
            f"Restaurants: {_compact(state.get('retrieved_restaurants', []), REST_KEYS)}\n"
            f"Recipes: {_compact(state.get('retrieved_recipes', []), RECIPE_KEYS)}"
        )

    def _analysis_node(self, persona: AgentPersona, key: str, instructions: str):
        def node(state: RecommendationState) -> dict:
            try:
                return {key: self._ask_json(persona, f"{self._analysis_context(state)}\n\n{instructions}")}
            except Exception as e:
                log.warning("%s failed: %s", persona.key, e)
                return {key: {}, "errors": [f"{persona.key}: {e}"]}

        return node

    def node_recommend(self, state: RecommendationState) -> dict:
        n = state.get("top_n", 3)
        prompt = f"""{self._analysis_context(state)}
Trend analysis: {json.dumps(state.get("trend_analysis", {}), ensure_ascii=False)}
Style analysis: {json.dumps(state.get("style_analysis", {}), ensure_ascii=False)}
Nutrition analysis: {json.dumps(state.get("nutrition_analysis", {}), ensure_ascii=False)}

Pick the top {n} restaurants and top {n} recipes FOR THIS USER. You MUST only choose names that
appear in the Restaurants / Recipes lists above, spelled exactly. Respect dietary restrictions.
Return JSON:
{{"summary": "one warm sentence addressed to the user",
  "restaurants": [{{"name": str, "reasoning": "2-3 sentences: profile fit, style, trend, nutrition"}}],
  "recipes": [{{"name": str, "reasoning": "2-3 sentences"}}]}}"""
        try:
            recs = self._ask_json(RECOMMENDER, prompt)
        except Exception as e:
            return {"final_recommendations": {}, "workflow_step": "failed", "errors": [f"recommend: {e}"]}

        # Grounding guardrail: keep only items that were actually retrieved, enriched with their facts.
        rest_by_name = {r["name"].lower(): r for r in state.get("retrieved_restaurants", [])}
        recipe_by_name = {r["name"].lower(): r for r in state.get("retrieved_recipes", [])}
        recs["restaurants"] = [
            {**rest_by_name[r["name"].lower()], "reasoning": r.get("reasoning", "")}
            for r in recs.get("restaurants", [])
            if isinstance(r, dict) and str(r.get("name", "")).lower() in rest_by_name
        ][:n]
        recs["recipes"] = [
            {**recipe_by_name[r["name"].lower()], "reasoning": r.get("reasoning", "")}
            for r in recs.get("recipes", [])
            if isinstance(r, dict) and str(r.get("name", "")).lower() in recipe_by_name
        ][:n]
        return {"final_recommendations": recs, "workflow_step": "complete"}

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(RecommendationState)
        g.add_node("profile", self.node_profile)
        g.add_node("retrieve", self.node_retrieve)
        g.add_node(
            "trends",
            self._analysis_node(
                TREND_ANALYST,
                "trend_analysis",
                "Identify 3 relevant food trends among these options. Return JSON: "
                '{"trends": [{"name": str, "description": str, "related_items": [str]}]}',
            ),
        )
        g.add_node(
            "styles",
            self._analysis_node(
                STYLE_EXPERT,
                "style_analysis",
                "For each option identify cuisine, regional variation, key cooking methods and flavor profile, "
                "and how well it matches the user. Return JSON: "
                '{"styles": [{"name": str, "flavor_profile": str, "cooking_methods": str, "match_reason": str}]}',
            ),
        )
        g.add_node(
            "nutrition",
            self._analysis_node(
                NUTRITION_EXPERT,
                "nutrition_analysis",
                "Check dietary restrictions, likely allergens and nutritional balance. Return JSON: "
                '{"compliant_items": [str], "flagged_items": [{"name": str, "reason": str}], '
                '"nutritional_highlights": [str]}',
            ),
        )
        g.add_node("recommend", self.node_recommend)

        g.add_edge(START, "profile")
        g.add_edge("profile", "retrieve")
        for parallel in ("trends", "styles", "nutrition"):
            g.add_edge("retrieve", parallel)
        g.add_edge(["trends", "styles", "nutrition"], "recommend")  # join: waits for all three
        g.add_edge("recommend", END)
        return g.compile()

    def run(self, user_input: str, top_n: int = 3) -> dict[str, Any]:
        return self.graph.invoke({"user_input": user_input, "top_n": top_n, "workflow_step": "start", "errors": []})


def format_recommendations(state: dict) -> str:
    """Markdown rendering for chat / UI."""
    recs = state.get("final_recommendations") or {}
    profile = state.get("user_profile") or {}
    lines: list[str] = []
    if recs.get("summary"):
        lines += [f"_{recs['summary']}_", ""]
    if profile.get("summary"):
        lines += [f"**Your dining profile:** {profile['summary']}", ""]
    if recs.get("restaurants"):
        lines.append("### 🍽️ Restaurants")
        for i, r in enumerate(recs["restaurants"], 1):
            facts = " · ".join(
                str(x)
                for x in (
                    r.get("cuisine"),
                    r.get("neighborhood"),
                    r.get("price_range"),
                    f"★ {r['rating']}" if r.get("rating") else None,
                )
                if x
            )
            lines += [f"**{i}. {r['name']}** — {facts}", f"{r.get('reasoning', '')}", ""]
    if recs.get("recipes"):
        lines.append("### 👩‍🍳 Cook at home")
        for i, r in enumerate(recs["recipes"], 1):
            meta = " · ".join(str(x) for x in (r.get("cuisine"), r.get("total_time")) if x)
            lines += [f"**{i}. {r['name']}** — {meta}", f"{r.get('reasoning', '')}", ""]
    trends = (state.get("trend_analysis") or {}).get("trends") or []
    if trends:
        lines.append("### 📈 Trends in your picks")
        lines += [f"- **{t.get('name')}**: {t.get('description')}" for t in trends if isinstance(t, dict)]
        lines.append("")
    flagged = (state.get("nutrition_analysis") or {}).get("flagged_items") or []
    if flagged:
        lines.append("### ⚠️ Dietary notes")
        lines += [f"- **{f.get('name')}**: {f.get('reason')}" if isinstance(f, dict) else f"- {f}" for f in flagged]
    if not (recs.get("restaurants") or recs.get("recipes")):
        lines.append("I couldn't build grounded recommendations this time — try adding more detail about your tastes.")
    return "\n".join(lines).strip()
