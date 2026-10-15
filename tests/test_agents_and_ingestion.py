# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : tests/test_agents_and_ingestion.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

import json

import pytest
from conftest import FakeLLM

from connoisseur.agents.workflow import RecommendationWorkflow, format_recommendations
from connoisseur.ingestion.extractor import EXTRACTION_SYSTEM, REPAIR_SYSTEM, RestaurantExtractor, split_culinary_map
from connoisseur.knowledge_base import KnowledgeBase
from connoisseur.llm import parse_json_object

VALID = json.dumps(
    {
        "name": "The Copper Sprout",
        "neighborhood": "Pasadena",
        "cuisine": "Modern Appalachian",
        "type": "farm-to-table",
        "rating": 4.4,
        "price_range": "$$",
        "signature_dish": "smoked trout",
        "vibes": ["cozy"],
        "description": "Reclaimed wood and amber lighting.",
        "shortcomings": [],
    }
)


def test_extractor_repairs_invalid_output():
    llm = FakeLLM({EXTRACTION_SYSTEM: "Sure! Here you go: {not json", REPAIR_SYSTEM: f"```json\n{VALID}\n```"})
    restaurant = RestaurantExtractor(llm).extract("The Copper Sprout is a ...")
    assert restaurant.name == "The Copper Sprout" and restaurant.price_level == 2
    assert sum(REPAIR_SYSTEM in c for c in llm.calls) == 1


def test_extractor_gives_up_after_bounded_repairs():
    llm = FakeLLM({EXTRACTION_SYSTEM: "{}", REPAIR_SYSTEM: '{"name": "only a name"}'})
    assert RestaurantExtractor(llm).extract("...") is None
    assert len(llm.calls) == 4  # 1 extraction + 3 repairs


def test_parse_json_object_tolerates_common_llm_glitches():
    # Granite often drops the outermost closing brace; braces inside strings must not confuse the repair
    assert parse_json_object('{"styles": [{"name": "Tacos {al pastor}"}]') == {"styles": [{"name": "Tacos {al pastor}"}]}
    assert parse_json_object('Sure! {"a": 1} Hope this helps {:)}') == {"a": 1}
    assert parse_json_object('```json\n{"a": [1, {"b": "cut off') == {"a": [1, {"b": "cut off"}]}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_split_culinary_map_drops_heading():
    paragraphs = split_culinary_map(KnowledgeBase().culinary_map())
    assert len(paragraphs) == 2 and paragraphs[0].startswith("**Iron & Embers**")


def _workflow_llm():
    return FakeLLM(
        {
            "User Profile Generator": json.dumps(
                {"favorite_cuisines": ["Japanese"], "dietary_restrictions": [], "summary": "A calm sushi lover."}
            ),
            "RAG Retriever": json.dumps(
                {"restaurant_queries": ["zen japanese"], "recipe_queries": ["noodles"], "max_price": None}
            ),
            "Food Trend Analyst": json.dumps({"trends": [{"name": "Omakase", "description": "Chef's choice."}]}),
            "Food Style Expert": json.dumps({"styles": [{"name": "Sakura Garden", "flavor_profile": "umami"}]}),
            "Nutrition Expert": json.dumps({"compliant_items": ["Sakura Garden"], "flagged_items": []}),
            "Recommendation Expert": json.dumps(
                {
                    "summary": "Enjoy!",
                    "restaurants": [
                        {"name": "Sakura Garden", "reasoning": "Serene and seasonal."},
                        {"name": "Hallucinated Bistro", "reasoning": "Does not exist."},
                    ],
                    "recipes": [{"name": "Spicy Dan Dan Noodles", "reasoning": "Bold."}],
                }
            ),
        }
    )


def test_workflow_runs_all_agents_and_grounds_output():
    llm = _workflow_llm()
    state = RecommendationWorkflow(llm_factory=lambda: llm).run("I love quiet Japanese dinners", top_n=3)

    assert state["workflow_step"] == "complete"
    assert state["errors"] == []
    assert state["trend_analysis"]["trends"] and state["style_analysis"] and state["nutrition_analysis"]
    names = [r["name"] for r in state["final_recommendations"]["restaurants"]]
    assert names == ["Sakura Garden"]  # hallucinated item removed by the grounding guardrail
    assert state["final_recommendations"]["restaurants"][0]["neighborhood"] == "Little Tokyo"  # enriched with facts
    assert len(llm.calls) == 6

    md = format_recommendations(state)
    assert "Sakura Garden" in md and "Hallucinated" not in md and "Spicy Dan Dan Noodles" in md


def test_workflow_degrades_gracefully_on_bad_json():
    llm = _workflow_llm()
    llm.rules["Food Trend Analyst"] = ["this is not json"]
    state = RecommendationWorkflow(llm_factory=lambda: llm).run("anything")
    assert state["workflow_step"] == "complete"
    assert any("food_trend_analyst" in e for e in state["errors"])
