# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/retrieval/vector_store.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Persistent ChromaDB index with three collections.

==================  ===========  =================================================
collection          encoder      content
==================  ===========  =================================================
``restaurants``     MiniLM 384d  every semantic attribute of each restaurant
``recipes``         MiniLM 384d  recipe name, cuisine, ingredients, image caption
``recipe_images``   CLIP 512d    the dish photos themselves
==================  ===========  =================================================

Numeric constraints (price, rating) are pushed down into Chroma ``where``
filters; free-text constraints (neighborhood, cuisine) are applied as
case-insensitive substring post-filters over an over-fetched candidate pool,
because users and LLMs rarely spell them exactly like the stored metadata.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterable
from pathlib import Path

from ..config import get_settings
from ..schemas import Recipe, Restaurant
from .embeddings import Embedder, get_embedder
from .fusion import Hit

log = logging.getLogger(__name__)

RESTAURANTS, RECIPES, IMAGES = "restaurants", "recipes", "recipe_images"
_COSINE = {"hnsw:space": "cosine"}  # distance = 1 - cosine similarity

# Chroma clients are not safe to create concurrently, so share one per path.
_CLIENTS: dict[str, object] = {}
_CLIENTS_LOCK = threading.Lock()


def _chroma_client(path: Path):
    key = str(path.resolve())
    with _CLIENTS_LOCK:
        if key not in _CLIENTS:
            import chromadb

            path.mkdir(parents=True, exist_ok=True)
            _CLIENTS[key] = chromadb.PersistentClient(path=key)
        return _CLIENTS[key]


def _restaurant_meta(r: Restaurant) -> dict:
    return {
        "name": r.name,
        "neighborhood": r.neighborhood,
        "cuisine": r.cuisine,
        "type": r.type,
        "rating": float(r.rating or 0.0),
        "price_range": r.price_range or "",
        "price_level": r.price_level,
        "vibes": ", ".join(r.vibes),
        "item_id": int(r.item_id or 0),
    }


def _recipe_meta(r: Recipe) -> dict:
    return {"name": r.name, "cuisine": r.cuisine, "recipe_id": r.id, "image": r.image_filename}


class VectorStore:
    def __init__(self, chroma_dir: Path | None = None, embedder: Embedder | None = None):
        self.chroma_dir = Path(chroma_dir or get_settings().chroma_dir)
        self._embedder = embedder
        self._client = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def client(self):
        if self._client is None:
            self._client = _chroma_client(self.chroma_dir)
        return self._client

    def _collection(self, name: str):
        return self.client.get_or_create_collection(name, metadata=_COSINE)

    # ----------------------------------------------------------------- status
    def counts(self) -> dict[str, int]:
        if not self.chroma_dir.exists():
            return {RESTAURANTS: 0, RECIPES: 0, IMAGES: 0}
        try:
            existing = {c if isinstance(c, str) else c.name for c in self.client.list_collections()}
            return {n: (self._collection(n).count() if n in existing else 0) for n in (RESTAURANTS, RECIPES, IMAGES)}
        except Exception as e:  # corrupt / incompatible index -> behave as "not built"
            log.warning("Vector index unavailable: %s", e)
            return {RESTAURANTS: 0, RECIPES: 0, IMAGES: 0}

    def has(self, name: str) -> bool:
        return self.counts().get(name, 0) > 0

    # --------------------------------------------------------------- building
    def reset(self, *names: str) -> None:
        for name in names:
            with contextlib.suppress(Exception):  # collection may not exist yet
                self.client.delete_collection(name)

    def upsert_restaurants(self, restaurants: Iterable[Restaurant]) -> int:
        items = [r for r in restaurants if r.name]
        if not items:
            return 0
        docs = [r.to_document() for r in items]
        self._collection(RESTAURANTS).upsert(
            ids=[f"rest_{r.item_id or i}" for i, r in enumerate(items)],
            embeddings=self.embedder.embed_texts(docs).tolist(),
            documents=docs,
            metadatas=[_restaurant_meta(r) for r in items],
        )
        return len(items)

    def delete_restaurant(self, restaurant: Restaurant) -> None:
        if self.has(RESTAURANTS):
            self._collection(RESTAURANTS).delete(ids=[f"rest_{restaurant.item_id}"])

    def upsert_recipes(self, recipes: Iterable[Recipe]) -> int:
        items = list(recipes)
        if not items:
            return 0
        docs = [r.to_document() for r in items]
        self._collection(RECIPES).upsert(
            ids=[f"recipe_{r.id}" for r in items],
            embeddings=self.embedder.embed_texts(docs).tolist(),
            documents=docs,
            metadatas=[_recipe_meta(r) for r in items],
        )
        return len(items)

    def upsert_recipe_images(self, recipes: Iterable[Recipe], images_dir: Path) -> int:
        # Pair each image with its recipe by id (recipe{id}.png), never by list position.
        pairs = [(r, images_dir / r.image_filename) for r in recipes]
        pairs = [(r, p) for r, p in pairs if p.exists()]
        if not pairs:
            return 0
        self._collection(IMAGES).upsert(
            ids=[f"img_{r.id}" for r, _ in pairs],
            embeddings=self.embedder.embed_images([p for _, p in pairs]).tolist(),
            documents=[r.name for r, _ in pairs],
            metadatas=[_recipe_meta(r) for r, _ in pairs],
        )
        return len(pairs)

    # -------------------------------------------------------------- searching
    def _query(self, name: str, vector, k: int, where: dict | None = None) -> list[Hit]:
        col = self._collection(name)
        n = col.count()
        if n == 0:
            return []
        res = col.query(
            query_embeddings=[vector.tolist()],
            n_results=min(k, n),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(id=i, document=d or "", metadata=m or {}, score=round(1.0 - float(dist), 4))
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
            )
        ]

    @staticmethod
    def _contains(hits: list[Hit], key: str, value: str | None) -> list[Hit]:
        if not value:
            return hits
        v = value.lower().strip()
        return [h for h in hits if v in str(h.metadata.get(key, "")).lower()]

    def search_restaurants(
        self,
        query: str,
        k: int = 5,
        neighborhood: str | None = None,
        cuisine: str | None = None,
        max_price_level: int | None = None,
        min_rating: float | None = None,
    ) -> list[Hit]:
        clauses = []
        if max_price_level:
            clauses.append({"price_level": {"$lte": int(max_price_level)}})
        if min_rating:
            clauses.append({"rating": {"$gte": float(min_rating)}})
        where = clauses[0] if len(clauses) == 1 else ({"$and": clauses} if clauses else None)

        pool = k * 4 if (neighborhood or cuisine) else k
        hits = self._query(RESTAURANTS, self.embedder.embed_texts([query])[0], pool, where)
        hits = self._contains(self._contains(hits, "neighborhood", neighborhood), "cuisine", cuisine)
        return hits[:k]

    def search_recipes(self, query: str, k: int = 5, cuisine: str | None = None) -> list[Hit]:
        pool = k * 4 if cuisine else k
        hits = self._query(RECIPES, self.embedder.embed_texts([query])[0], pool)
        return self._contains(hits, "cuisine", cuisine)[:k]

    def search_images_by_text(self, query: str, k: int = 5, cuisine: str | None = None) -> list[Hit]:
        pool = k * 4 if cuisine else k
        hits = self._query(IMAGES, self.embedder.embed_clip_texts([query])[0], pool)
        return self._contains(hits, "cuisine", cuisine)[:k]

    def similar_dishes(self, recipe_id: int, k: int = 5, same_cuisine: bool = False) -> list[Hit]:
        """Image -> image search, reusing the stored CLIP vector of the query dish."""
        col = self._collection(IMAGES)
        got = col.get(ids=[f"img_{recipe_id}"], include=["embeddings", "metadatas"])
        if not got["ids"]:
            return []
        import numpy as np

        vector = np.asarray(got["embeddings"][0], dtype=np.float32)
        where = {"cuisine": got["metadatas"][0]["cuisine"]} if same_cuisine else None
        hits = self._query(IMAGES, vector, k + 1, where)
        return [h for h in hits if h.id != f"img_{recipe_id}"][:k]


def build_index(kb, store: VectorStore | None = None, include_images: bool = True) -> dict[str, int]:
    """(Re)build every collection from the knowledge base."""
    store = store or VectorStore()
    store.reset(RESTAURANTS, RECIPES, IMAGES)
    counts = {
        RESTAURANTS: store.upsert_restaurants(kb.restaurants()),
        RECIPES: store.upsert_recipes(kb.recipes()),
        IMAGES: 0,
    }
    if include_images:
        images_dir = get_settings().images_dir
        if images_dir.exists():
            counts[IMAGES] = store.upsert_recipe_images(kb.recipes(), images_dir)
        else:
            log.warning("No images at %s - run `connoisseur download-data --with-images` first", images_dir)
    return counts
