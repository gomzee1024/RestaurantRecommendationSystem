# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/knowledge_base.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""JSON-backed knowledge base: restaurants, reviews and recipes.

Writes are atomic and keep a ``.bak`` of the previous version, so a bad edit
can always be rolled back.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from .config import get_settings
from .schemas import Recipe, Restaurant, Review

FIRST_ITEM_ID = 1_000_001


def _read_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy(path, path.with_suffix(path.suffix + ".bak"))
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("&", "and")).strip()


class KnowledgeBase:
    def __init__(
        self,
        restaurants_path: Path | None = None,
        reviews_path: Path | None = None,
        recipes_path: Path | None = None,
        culinary_map_path: Path | None = None,
    ):
        s = get_settings()
        self.restaurants_path = restaurants_path or s.restaurants_path
        self.reviews_path = reviews_path or s.reviews_path
        self.recipes_path = recipes_path or s.recipes_path
        self.culinary_map_path = culinary_map_path or s.culinary_map_path

    # ------------------------------------------------------------------ reads
    # Files are re-read on every call so edits made by another process (CLI,
    # UI, ingestion) are visible to a long-running MCP server immediately.
    def restaurants(self) -> list[Restaurant]:
        return [Restaurant.model_validate(r) for r in _read_json(self.restaurants_path)]

    def reviews(self) -> list[Review]:
        return [Review.model_validate(r) for r in _read_json(self.reviews_path)]

    def recipes(self) -> list[Recipe]:
        return [Recipe.model_validate(r) for r in _read_json(self.recipes_path)]

    def culinary_map(self) -> str:
        return self.culinary_map_path.read_text(encoding="utf-8") if self.culinary_map_path.exists() else ""

    def find_restaurants(self, name: str) -> list[Restaurant]:
        """Bidirectional partial name match, so 'Iron' finds 'Iron & Embers'."""
        q = _norm(name)
        if not q:
            return []
        return [r for r in self.restaurants() if q in _norm(r.name) or _norm(r.name) in q]

    def find_by_vibe(self, vibe: str) -> list[Restaurant]:
        v = vibe.lower().strip()
        return [r for r in self.restaurants() if any(v in tag.lower() for tag in r.vibes) or v in r.description.lower()]

    def culinary_map_excerpts(self, keyword: str, limit: int = 5, max_chars: int = 300) -> list[str]:
        k = keyword.lower().strip()
        paragraphs = [p.strip() for p in self.culinary_map().split("\n\n") if p.strip()]
        return [p[:max_chars] for p in paragraphs if k in p.lower()][:limit]

    def find_review(self, restaurant_name: str) -> Review | None:
        q = _norm(restaurant_name)
        for review in self.reviews():
            if q in _norm(review.restaurant_name) or _norm(review.restaurant_name) in q:
                return review
        return None

    def find_recipe(self, name: str) -> Recipe | None:
        q = _norm(name)
        recipes = self.recipes()
        exact = [r for r in recipes if _norm(r.name) == q]
        if exact:
            return exact[0]
        partial = [r for r in recipes if q in _norm(r.name)]
        return partial[0] if partial else None

    def keyword_search(self, query: str, k: int = 5) -> list[Restaurant]:
        """Dependency-free fallback ranking used when the vector index is not built."""
        terms = [t for t in re.findall(r"[a-z]+", query.lower()) if len(t) > 2]
        scored = []
        for r in self.restaurants():
            doc = r.to_document().lower()
            score = sum(doc.count(t) for t in terms)
            if score:
                scored.append((score, r.rating or 0, r))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [r for _, _, r in scored[:k]]

    # ----------------------------------------------------------------- writes
    def _save_restaurants(self, items: list[Restaurant]) -> None:
        _write_json_atomic(self.restaurants_path, [r.model_dump(exclude_none=False) for r in items])

    def next_item_id(self, items: list[Restaurant] | None = None) -> int:
        items = self.restaurants() if items is None else items
        ids = [r.item_id for r in items if r.item_id]
        return max(ids) + 1 if ids else FIRST_ITEM_ID

    def add_restaurant(self, restaurant: Restaurant) -> Restaurant:
        items = self.restaurants()
        if any(_norm(r.name) == _norm(restaurant.name) for r in items):
            raise ValueError(f"'{restaurant.name}' already exists in the knowledge base")
        restaurant.item_id = restaurant.item_id or self.next_item_id(items)
        items.append(restaurant)
        self._save_restaurants(items)
        return restaurant

    def update_restaurant(self, name: str, **changes) -> Restaurant:
        items = self.restaurants()
        for i, r in enumerate(items):
            if _norm(r.name) == _norm(name):
                updated = Restaurant.model_validate({**r.model_dump(), **changes, "item_id": r.item_id})
                items[i] = updated
                self._save_restaurants(items)
                return updated
        raise KeyError(f"No restaurant named '{name}'")

    def delete_restaurant(self, name: str) -> Restaurant:
        items = self.restaurants()
        for i, r in enumerate(items):
            if _norm(r.name) == _norm(name):
                removed = items.pop(i)
                self._save_restaurants(items)
                return removed
        raise KeyError(f"No restaurant named '{name}'")

    def save_recipes(self, recipes: list[Recipe]) -> None:
        _write_json_atomic(self.recipes_path, [r.model_dump() for r in recipes])
