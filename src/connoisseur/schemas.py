# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/schemas.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Pydantic schemas for every record in the knowledge base.

``Restaurant`` is also the contract the LLM extraction pipeline must satisfy:
its JSON schema is fed to the auto-repair prompt when validation fails.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Restaurant(BaseModel):
    name: str
    neighborhood: str
    cuisine: str
    type: str = Field(description="Kind of venue, e.g. 'upscale bistro', 'casual taqueria'")
    rating: float | None = Field(default=None, ge=0, le=5)
    price_range: str | None = Field(default=None, description="'$' to '$$$$'")
    signature_dish: str | None = None
    vibes: list[str] = Field(default_factory=list)
    description: str = ""
    shortcomings: list[str] = Field(default_factory=list)
    item_id: int | None = None

    @field_validator("price_range", mode="before")
    @classmethod
    def _normalise_price(cls, v):
        """Accept 3, "3" or "$$$" and store the canonical dollar-sign form."""
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().isdigit()):
            n = max(1, min(4, int(float(v))))
            return "$" * n
        dollars = str(v).count("$")
        return "$" * max(1, min(4, dollars)) if dollars else str(v)

    @field_validator("vibes", "shortcomings", mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @property
    def price_level(self) -> int:
        return len(self.price_range) if self.price_range and set(self.price_range) == {"$"} else 0

    def to_document(self) -> str:
        """Text used for embedding: every attribute that carries semantic signal."""
        parts = [
            f"Restaurant: {self.name}",
            f"Neighborhood: {self.neighborhood}",
            f"Cuisine: {self.cuisine}",
            f"Type: {self.type}",
        ]
        if self.vibes:
            parts.append(f"Vibes: {', '.join(self.vibes)}")
        if self.signature_dish:
            parts.append(f"Signature dish: {self.signature_dish}")
        if self.description:
            parts.append(f"Description: {self.description}")
        return "\n".join(parts)


class Review(BaseModel):
    restaurant_name: str
    reviewer: str
    rating: float | None = None
    review_text: str
    image_description: str | None = None
    visit_date: str | None = None


class Recipe(BaseModel):
    id: int
    name: str
    cuisine: str
    servings: int | None = None
    prep_time: str | None = None
    cook_time: str | None = None
    total_time: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    image_description: str | None = None

    @property
    def image_filename(self) -> str:
        return f"recipe{self.id}.png"

    def to_document(self) -> str:
        parts = [f"Recipe: {self.name}", f"Cuisine: {self.cuisine}"]
        if self.ingredients:
            parts.append(f"Ingredients: {'; '.join(self.ingredients)}")
        if self.image_description:
            parts.append(f"Looks like: {self.image_description}")
        return "\n".join(parts)
