# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/retrieval/fusion.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Cross-modal score calibration and weighted fusion.

Cosine scores from MiniLM and CLIP live on different scales, so each list is
min-max normalised to [0, 1] before applying ``w_text * s_text + w_img * s_img``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Hit:
    id: str
    document: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0  # cosine similarity (higher is better)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FusedHit:
    modality: str  # "restaurant" | "dish_image"
    id: str
    title: str
    metadata: dict
    text_score: float
    image_score: float
    fused_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def minmax(scores: Sequence[float]) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.ones_like(x)  # all equal -> same confidence
    return (x - lo) / (hi - lo)


def fuse(
    text_hits: Sequence[Hit],
    image_hits: Sequence[Hit],
    w_text: float = 0.6,
    w_img: float = 0.4,
    top_n: int | None = 5,
) -> list[FusedHit]:
    t_norm, i_norm = minmax([h.score for h in text_hits]), minmax([h.score for h in image_hits])
    rows = [
        FusedHit("restaurant", h.id, h.metadata.get("name", h.id), h.metadata, float(s), 0.0, float(w_text * s))
        for h, s in zip(text_hits, t_norm, strict=True)
    ] + [
        FusedHit("dish_image", h.id, h.metadata.get("name", h.id), h.metadata, 0.0, float(s), float(w_img * s))
        for h, s in zip(image_hits, i_norm, strict=True)
    ]
    rows.sort(key=lambda r: r.fused_score, reverse=True)
    return rows if top_n is None else rows[: max(0, top_n)]
