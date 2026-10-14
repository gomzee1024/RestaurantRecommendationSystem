# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : tests/test_fusion_and_retrieval.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

import numpy as np
import pytest

from connoisseur.retrieval.fusion import Hit, fuse, minmax
from connoisseur.retrieval.service import RetrievalService


def test_minmax_handles_edge_cases():
    assert minmax([]).size == 0
    assert np.allclose(minmax([0.3, 0.3]), [1.0, 1.0])
    assert np.allclose(minmax([0.2, 0.6, 0.4]), [0.0, 1.0, 0.5])


def test_fusion_respects_weights():
    text = [Hit("rest_1", "", {"name": "A"}, 0.9), Hit("rest_2", "", {"name": "B"}, 0.5)]
    imgs = [Hit("img_1", "", {"name": "Pizza"}, 0.31), Hit("img_2", "", {"name": "Tacos"}, 0.25)]

    text_heavy = fuse(text, imgs, w_text=0.8, w_img=0.2, top_n=None)
    assert [r.title for r in text_heavy][:2] == ["A", "Pizza"]
    assert text_heavy[0].fused_score == pytest.approx(0.8)

    image_heavy = fuse(text, imgs, w_text=0.3, w_img=0.7, top_n=2)
    assert [r.title for r in image_heavy] == ["Pizza", "A"]
    assert image_heavy[0].modality == "dish_image"


def test_keyword_fallback_with_filters():
    svc = RetrievalService()  # no index built in the test data dir -> keyword mode
    res = svc.search_restaurants("moody steakhouse")
    assert res["mode"] == "keyword"
    assert res["results"][0]["name"] == "Iron & Embers"

    cheap = svc.search_restaurants("seafood tacos dining", max_price="$$")
    assert [r["name"] for r in cheap["results"]] == ["Mar de Cortez"]

    assert svc.search_restaurants("steakhouse", neighborhood="Little Tokyo")["results"] == []


def test_recipe_fallback_and_unavailable_multimodal():
    svc = RetrievalService()
    assert svc.search_recipes("spicy noodles")["results"][0]["name"] == "Spicy Dan Dan Noodles"
    assert svc.multimodal_search("sushi")["status"] == "unavailable"
    assert svc.similar_dishes("does not exist")["status"] == "not_found"
