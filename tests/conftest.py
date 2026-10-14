# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : tests/conftest.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Test fixtures: an isolated data directory and a scriptable fake LLM (no API keys needed)."""

import json
import os
import tempfile
from pathlib import Path

# Point the app at a throw-away data dir *before* any connoisseur module reads settings.
_DATA = Path(tempfile.mkdtemp(prefix="connoisseur-test-"))
os.environ["DATA_DIR"] = str(_DATA)
os.environ["LLM_PROVIDER"] = "watsonx"

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

RESTAURANTS = [
    {
        "name": "Iron & Embers",
        "neighborhood": "Arts District, DTLA",
        "cuisine": "American steakhouse",
        "type": "fine dining",
        "rating": 4.8,
        "price_range": "$$$$",
        "signature_dish": "45-day dry-aged ribeye",
        "vibes": ["moody", "industrial"],
        "description": "Smells deeply of white oak smoke.",
        "item_id": 1000001,
    },
    {
        "name": "Sakura Garden",
        "neighborhood": "Little Tokyo",
        "cuisine": "Japanese Kaiseki-lite",
        "type": "intimate",
        "rating": 4.9,
        "price_range": "$$$",
        "signature_dish": "bluefin tuna belly",
        "vibes": ["zen", "minimalist"],
        "description": "A quiet escape focused on seasonal perfection.",
        "item_id": 1000002,
    },
    {
        "name": "Mar de Cortez",
        "neighborhood": "Santa Monica",
        "cuisine": "Baja-style seafood",
        "type": "casual taqueria",
        "rating": 4.2,
        "price_range": "$",
        "signature_dish": "snapper tacos",
        "vibes": ["sun-drenched", "casual"],
        "description": "Open-air dining near the pier.",
        "item_id": 1000003,
    },
]
REVIEWS = [
    {
        "restaurant_name": "Iron & Embers",
        "reviewer": "SteakFan",
        "rating": 4.8,
        "review_text": "The ribeye was perfect and the room is dark and moody.",
        "image_description": "A charred ribeye on a cast-iron board.",
        "visit_date": "2024-10-01",
    }
]
RECIPES = [
    {"id": 1, "name": "Classic Margherita Pizza", "cuisine": "Italian", "ingredients": ["dough", "tomato", "basil"]},
    {
        "id": 2,
        "name": "Spicy Dan Dan Noodles",
        "cuisine": "Chinese (Szechuan)",
        "ingredients": ["noodles", "chili oil"],
    },
    {"id": 3, "name": "Vegan Buddha Bowl", "cuisine": "American", "ingredients": ["quinoa", "chickpeas", "kale"]},
]
CULINARY_MAP = """### The Culinary Map of California

**Iron & Embers** anchors the **Arts District** as a **moody** steakhouse. Price range: $$$$

**Sakura Garden** in **Little Tokyo** offers a **zen** Kaiseki-lite menu. Price range: $$$
"""


def write_fixture_data(data_dir: Path) -> None:
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "restaurants.json").write_text(json.dumps(RESTAURANTS), encoding="utf-8")
    (data_dir / "reviews.json").write_text(json.dumps(REVIEWS), encoding="utf-8")
    (data_dir / "recipes.json").write_text(json.dumps(RECIPES), encoding="utf-8")
    (data_dir / "raw" / "California-Culinary-Map.txt").write_text(CULINARY_MAP, encoding="utf-8")


@pytest.fixture(autouse=True)
def fresh_data():
    """Every test starts from the same pristine fixture files."""
    for f in _DATA.glob("*.json*"):
        f.unlink()
    write_fixture_data(_DATA)
    yield _DATA


class FakeLLM:
    """Duck-typed chat model: picks a reply by matching the system prompt."""

    def __init__(self, rules: dict[str, str | list[str]], default: str = "{}"):
        self.rules = {k: (v if isinstance(v, list) else [v]) for k, v in rules.items()}
        self.default = default
        self.calls: list[str] = []

    def invoke(self, messages):
        system = messages[0].content
        self.calls.append(system)
        for needle, replies in self.rules.items():
            if needle in system:
                return AIMessage(content=replies.pop(0) if len(replies) > 1 else replies[0])
        return AIMessage(content=self.default)
