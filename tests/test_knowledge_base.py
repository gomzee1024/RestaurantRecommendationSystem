# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : tests/test_knowledge_base.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

import pytest

from connoisseur.knowledge_base import KnowledgeBase
from connoisseur.schemas import Restaurant


def test_price_and_list_normalisation():
    r = Restaurant(name="X", neighborhood="Y", cuisine="Z", type="T", price_range=3, vibes="cozy, warm")
    assert r.price_range == "$$$"
    assert r.price_level == 3
    assert r.vibes == ["cozy", "warm"]
    assert Restaurant(name="X", neighborhood="Y", cuisine="Z", type="T", price_range="$$").price_level == 2


def test_partial_name_lookup_is_bidirectional():
    kb = KnowledgeBase()
    assert [r.name for r in kb.find_restaurants("Iron")] == ["Iron & Embers"]
    assert [r.name for r in kb.find_restaurants("tell me about sakura garden please")] == ["Sakura Garden"]
    assert kb.find_restaurants("Nonexistent") == []


def test_vibe_and_review_lookup():
    kb = KnowledgeBase()
    assert {r.name for r in kb.find_by_vibe("moody")} == {"Iron & Embers"}
    assert kb.find_review("iron and embers").reviewer == "SteakFan"
    assert kb.find_review("Sakura") is None
    assert kb.culinary_map_excerpts("zen")[0].startswith("**Sakura Garden**")


def test_keyword_search_ranks_relevant_first():
    assert KnowledgeBase().keyword_search("zen japanese tuna")[0].name == "Sakura Garden"


def test_crud_with_backup(fresh_data):
    kb = KnowledgeBase()
    new = kb.add_restaurant(
        Restaurant(name="The Copper Sprout", neighborhood="Pasadena", cuisine="Appalachian", type="bistro")
    )
    assert new.item_id == 1000004
    assert (fresh_data / "restaurants.json.bak").exists()

    with pytest.raises(ValueError):
        kb.add_restaurant(Restaurant(name="the copper sprout", neighborhood="a", cuisine="b", type="c"))

    assert kb.update_restaurant("The Copper Sprout", rating=4.6).rating == 4.6
    assert kb.delete_restaurant("The Copper Sprout").name == "The Copper Sprout"
    assert len(kb.restaurants()) == 3
    with pytest.raises(KeyError):
        kb.delete_restaurant("The Copper Sprout")
