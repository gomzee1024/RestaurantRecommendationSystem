# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/ingestion/extractor.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Unstructured restaurant description -> validated ``Restaurant`` record.

Pipeline: one-shot extraction prompt -> Pydantic validation -> if invalid, an
LLM "repair expert" receives the broken output, the validation error and the
target JSON schema, and tries again (bounded number of attempts).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from ..llm import get_chat_model, message_text
from ..schemas import Restaurant

log = logging.getLogger(__name__)

MAX_REPAIR_ATTEMPTS = 3

EXAMPLE_PARAGRAPH = (
    "Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** "
    "specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of "
    "the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a "
    "premier spot for open-air dining near the pier. Price range: $"
)
EXAMPLE_OUTPUT = {
    "name": "Mar de Cortez",
    "neighborhood": "Santa Monica",
    "cuisine": "Baja-style seafood",
    "type": "casual taqueria",
    "rating": 4.2,
    "price_range": "$",
    "signature_dish": "beer-battered snapper tacos and zesty octopus ceviche",
    "vibes": ["sun-drenched", "casual"],
    "description": "Captures the salt-air energy of the coast, a premier spot for open-air dining near the pier.",
    "shortcomings": [],
}

EXTRACTION_SYSTEM = (
    "You are a meticulous data extraction engine. You convert restaurant descriptions into "
    "clean JSON that follows the requested schema exactly. You never invent facts that are not "
    "in the description, and you output JSON only."
)

REPAIR_SYSTEM = "You are an expert at repairing malformed JSON so that it validates against a JSON schema."


def extraction_prompt(paragraph: str) -> str:
    return f"""Extract the restaurant described below into a single JSON object.

Rules:
- Use exactly these keys: {", ".join(EXAMPLE_OUTPUT)}.
- "price_range" is the dollar signs from the text (e.g. "$$$"); null if absent.
- "rating" is a number out of 5; null if absent.
- "vibes" is a list of short atmosphere keywords; "shortcomings" lists any negatives mentioned.
- Return ONLY the JSON object, no markdown and no commentary.

Example input:
{EXAMPLE_PARAGRAPH}

Example output:
{json.dumps(EXAMPLE_OUTPUT, indent=2)}

Restaurant description:
{paragraph}
"""


def repair_prompt(candidate: str, error: str) -> str:
    return f"""The following output failed validation. Fix it.

Candidate output:
{candidate}

Validation error:
{error}

Required JSON schema:
{json.dumps(Restaurant.model_json_schema())}

Return ONLY the corrected raw JSON object with exactly the schema's field names. No explanations, no markdown."""


def _clip_to_object(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class RestaurantExtractor:
    def __init__(self, llm: BaseChatModel | None = None):
        self.llm = llm or get_chat_model(temperature=0)

    def _ask(self, system: str, prompt: str) -> str:
        reply = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        return message_text(reply.content)

    def extract(self, paragraph: str) -> Restaurant | None:
        candidate = self._ask(EXTRACTION_SYSTEM, extraction_prompt(paragraph))
        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            candidate = _clip_to_object(candidate)
            try:
                return Restaurant.model_validate_json(candidate)
            except ValidationError as e:
                if attempt == MAX_REPAIR_ATTEMPTS:
                    log.warning("Still invalid after %d repair attempts; skipping", MAX_REPAIR_ATTEMPTS)
                    return None
                log.debug("Validation failed (attempt %d): %s", attempt + 1, e)
                candidate = self._ask(REPAIR_SYSTEM, repair_prompt(candidate, e.json()))
        return None


def split_culinary_map(text: str) -> list[str]:
    """One paragraph per restaurant; drops the markdown heading."""
    return [p.strip() for p in text.split("\n\n") if p.strip() and not p.lstrip().startswith("#")]


def extract_many(paragraphs: Iterable[str], extractor: RestaurantExtractor) -> Iterator[tuple[int, Restaurant | None]]:
    for i, paragraph in enumerate(paragraphs):
        yield i, extractor.extract(paragraph)
