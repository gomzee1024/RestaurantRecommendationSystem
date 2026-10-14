# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/ingestion/captioner.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Vision-LLM image captioning so images become searchable text.

Works with local files (recipe photos) and URLs (user review photos); the
optional ``context`` grounds the caption in what a reviewer wrote.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from ..llm import get_vision_model, message_text

DISH_SYSTEM = (
    "You are a professional food photographer and culinary expert. You describe food images "
    "accurately and concisely, based only on what is visible."
)
REVIEW_SYSTEM = (
    "You are a culinary expert. You write concise image descriptions that combine what is visible "
    "in a food photo with the sentiment and details expressed in the customer's review."
)


def dish_prompt(food_name: str) -> str:
    return (
        f'Describe this image of "{food_name}" in 2-3 sentences. Focus on the visible ingredients, '
        "cooking style, colors, plating and presentation. Do not speculate about taste, origin or "
        "anything that is not visible in the image. Return plain text only."
    )


def review_prompt(review_text: str) -> str:
    return (
        f'Here is a customer review of a restaurant visit:\n"""{review_text}"""\n\n'
        "Describe the attached image in 2-3 sentences. Describe the food and presentation you actually "
        "see, and align the description with the sentiment and details mentioned in the review. Do not "
        "invent details that are neither visible nor supported by the review. Return plain text only."
    )


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
def _fetch(url: str) -> tuple[bytes, str]:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.content, r.headers.get("content-type", "image/jpeg").split(";")[0]


def to_data_uri(source: str | Path) -> str:
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        data, mime = _fetch(source)
    else:
        path = Path(source)
        data = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


class ImageCaptioner:
    def __init__(self, llm: BaseChatModel | None = None):
        self.llm = llm or get_vision_model()

    def caption(self, image: str | Path, system: str, prompt: str) -> str:
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": to_data_uri(image)}},
            ]
        )
        reply = self.llm.invoke([SystemMessage(content=system), message])
        return message_text(reply.content).strip()

    def caption_dish(self, image: str | Path, food_name: str) -> str:
        return self.caption(image, DISH_SYSTEM, dish_prompt(food_name))

    def caption_review_image(self, image: str | Path, review_text: str) -> str:
        return self.caption(image, REVIEW_SYSTEM, review_prompt(review_text))
