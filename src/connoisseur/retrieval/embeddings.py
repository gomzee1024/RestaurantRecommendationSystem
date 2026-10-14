# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/retrieval/embeddings.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Text (MiniLM, 384-d) and image/text (CLIP, 512-d) encoders.

Models are loaded lazily on first use and cached for the life of the process,
so the MCP server starts instantly and only pays the load cost once. All
vectors are L2-normalised, which makes cosine similarity a dot product.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..config import get_settings


class Embedder:
    def __init__(self, text_model: str | None = None, clip_model: str | None = None, device: str = "cpu"):
        s = get_settings()
        self.text_model_name = text_model or s.text_embedding_model
        self.clip_model_name = clip_model or s.clip_model
        self.device = device
        self._text = None
        self._clip = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- loaders
    def _text_model(self):
        with self._lock:
            if self._text is None:
                from sentence_transformers import SentenceTransformer

                self._text = SentenceTransformer(self.text_model_name, device=self.device)
        return self._text

    def _clip_parts(self):
        with self._lock:
            if self._clip is None:
                from transformers import CLIPModel, CLIPProcessor

                model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device).eval()
                processor = CLIPProcessor.from_pretrained(self.clip_model_name)
                self._clip = (model, processor)
        return self._clip

    # --------------------------------------------------------------- encoders
    def embed_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return (
            self._text_model()
            .encode(list(texts), batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
            .astype(np.float32)
        )

    def embed_images(self, paths: Sequence[str | Path], batch_size: int = 16) -> np.ndarray:
        import torch
        from PIL import Image

        model, processor = self._clip_parts()
        out = []
        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                images = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
                inputs = processor(images=images, return_tensors="pt").to(self.device)
                out.append(_normalise(_as_tensor(model.get_image_features(**inputs))))
        return np.vstack(out)

    def embed_clip_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text into CLIP's joint space, enabling text -> image retrieval."""
        import torch

        model, processor = self._clip_parts()
        with torch.no_grad():
            inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
            return _normalise(_as_tensor(model.get_text_features(**inputs)))


def _as_tensor(features):
    # transformers>=5 may return a model output object instead of a bare tensor
    if hasattr(features, "pooler_output") and not hasattr(features, "norm"):
        return features.pooler_output
    return features


def _normalise(t) -> np.ndarray:
    t = t / t.norm(dim=-1, keepdim=True)
    return t.cpu().numpy().astype(np.float32)


_default: Embedder | None = None


def get_embedder() -> Embedder:
    global _default
    if _default is None:
        _default = Embedder()
    return _default
