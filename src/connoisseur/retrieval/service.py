# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/retrieval/service.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Single retrieval façade shared by the MCP tools and the agent workflow.

Uses the vector index when it has been built and transparently degrades to
keyword search over the JSON knowledge base when it has not, so the app is
usable even on a machine without the embedding models.
"""

from __future__ import annotations

import logging
import re

from ..knowledge_base import KnowledgeBase
from ..schemas import Restaurant
from .fusion import fuse
from .vector_store import IMAGES, RECIPES, RESTAURANTS, VectorStore

log = logging.getLogger(__name__)


def _price_level(max_price: str | int | None) -> int | None:
    if max_price in (None, "", 0):
        return None
    if isinstance(max_price, int):
        return max_price
    s = str(max_price).strip()
    return int(s) if s.isdigit() else (s.count("$") or None)


class RetrievalService:
    def __init__(self, kb: KnowledgeBase | None = None, store: VectorStore | None = None):
        self.kb = kb or KnowledgeBase()
        self.store = store or VectorStore()

    def status(self) -> dict:
        return {"vector_index": self.store.counts(), "restaurants_in_kb": len(self.kb.restaurants())}

    # ------------------------------------------------------------ restaurants
    def search_restaurants(
        self,
        query: str,
        k: int = 5,
        neighborhood: str | None = None,
        cuisine: str | None = None,
        max_price: str | int | None = None,
        min_rating: float | None = None,
    ) -> dict:
        level = _price_level(max_price)
        by_name = {r.name: r for r in self.kb.restaurants()}

        if self.store.has(RESTAURANTS):
            hits = self.store.search_restaurants(query, k, neighborhood, cuisine, level, min_rating)
            results = [
                {**by_name[h.metadata["name"]].model_dump(exclude={"shortcomings"}), "similarity": h.score}
                for h in hits
                if h.metadata.get("name") in by_name
            ]
            return {"mode": "semantic", "results": results}

        candidates = self.kb.keyword_search(query, k=len(by_name)) or list(by_name.values())
        results = [r for r in candidates if self._passes(r, neighborhood, cuisine, level, min_rating)][:k]
        return {"mode": "keyword", "results": [r.model_dump(exclude={"shortcomings"}) for r in results]}

    @staticmethod
    def _passes(r: Restaurant, neighborhood, cuisine, level, min_rating) -> bool:
        if neighborhood and neighborhood.lower() not in r.neighborhood.lower():
            return False
        if cuisine and cuisine.lower() not in r.cuisine.lower():
            return False
        if level and r.price_level and r.price_level > level:
            return False
        return not (min_rating and (r.rating or 0) < min_rating)

    # ---------------------------------------------------------------- recipes
    def search_recipes(self, query: str, k: int = 5, cuisine: str | None = None) -> dict:
        recipes = {f"recipe_{r.id}": r for r in self.kb.recipes()}
        if self.store.has(RECIPES):
            hits = self.store.search_recipes(query, k, cuisine)
            return {
                "mode": "semantic",
                "results": [self._recipe_card(recipes[h.id], h.score) for h in hits if h.id in recipes],
            }
        terms = [t for t in re.findall(r"[a-z]+", query.lower()) if len(t) > 2]
        scored = []
        for r in recipes.values():
            if cuisine and cuisine.lower() not in r.cuisine.lower():
                continue
            doc = r.to_document().lower()
            score = sum(doc.count(t) for t in terms)
            if score:
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return {"mode": "keyword", "results": [self._recipe_card(r) for _, r in scored[:k]]}

    @staticmethod
    def _recipe_card(r, score: float | None = None) -> dict:
        card = {
            "name": r.name,
            "cuisine": r.cuisine,
            "total_time": r.total_time,
            "servings": r.servings,
            "ingredients": r.ingredients,
        }
        if r.image_description:
            card["image_description"] = r.image_description
        if score is not None:
            card["similarity"] = score
        return card

    # -------------------------------------------------------------- multimodal
    def multimodal_search(self, query: str, k: int = 5, text_weight: float = 0.6, image_weight: float = 0.4) -> dict:
        if not (self.store.has(RESTAURANTS) and self.store.has(IMAGES)):
            return {
                "status": "unavailable",
                "message": "Multimodal search needs the vector index with images. "
                "Run `connoisseur download-data --with-images` then `connoisseur build-index`.",
            }
        text_hits = self.store.search_restaurants(query, k)
        image_hits = self.store.search_images_by_text(query, k)
        fused = fuse(text_hits, image_hits, text_weight, image_weight, top_n=k)
        return {
            "status": "ok",
            "weights": {"text": text_weight, "image": image_weight},
            "results": [
                {
                    "modality": f.modality,
                    "name": f.title,
                    "cuisine": f.metadata.get("cuisine"),
                    "neighborhood": f.metadata.get("neighborhood"),
                    "fused_score": round(f.fused_score, 4),
                    "text_score": round(f.text_score, 4),
                    "image_score": round(f.image_score, 4),
                }
                for f in fused
            ],
        }

    def similar_dishes(self, recipe_name: str, k: int = 5, same_cuisine: bool = False) -> dict:
        recipe = self.kb.find_recipe(recipe_name)
        if recipe is None:
            return {"status": "not_found", "message": f"No recipe matching '{recipe_name}'."}
        if not self.store.has(IMAGES):
            return {"status": "unavailable", "message": "Image index not built (see `connoisseur build-index`)."}
        hits = self.store.similar_dishes(recipe.id, k, same_cuisine)
        return {
            "status": "ok",
            "query_dish": {"name": recipe.name, "cuisine": recipe.cuisine},
            "visually_similar": [
                {"name": h.metadata.get("name"), "cuisine": h.metadata.get("cuisine"), "similarity": h.score}
                for h in hits
            ],
        }
