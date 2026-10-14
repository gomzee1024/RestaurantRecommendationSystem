# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/ingestion/download.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Fetch the public IBM Skills Network capstone datasets into ``data/``."""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings
from ..knowledge_base import FIRST_ITEM_ID
from ..schemas import Restaurant

log = logging.getLogger(__name__)

_BASE = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud"
SOURCES = {
    "California-Culinary-Map.txt": f"{_BASE}/_nbA_KMj1n7yBrpfz8rYkg/California-Culinary-Map.txt",
    "structured-restaurant-data.json": f"{_BASE}/lxfhTQUrDCCD_JSMmr92VA/structured-restaurant-data.json",
    "augmented-user-review.json": f"{_BASE}/oMqDIzTBNFT7KKJ0GW4-Cw/augmented-user-review.json",
    "Recipes.json": f"{_BASE}/hpTjb6liKBLVHQK0UgMi5A/Recipes.json",
}
IMAGES_ZIP = f"{_BASE}/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"  # ~215 MB


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.replace(dest)


def download_all(with_images: bool = False, force: bool = False) -> None:
    s = get_settings()
    for filename, url in SOURCES.items():
        dest = s.raw_dir / filename
        if dest.exists() and not force:
            log.info("✓ %s (cached)", filename)
            continue
        log.info("↓ %s", filename)
        _download(url, dest)

    if with_images:
        if s.images_dir.exists() and any(s.images_dir.glob("*.png")) and not force:
            log.info("✓ recipe images (cached)")
        else:
            zip_path = s.raw_dir / "synthetic-recipe-images.zip"
            if not zip_path.exists() or force:
                log.info("↓ recipe images (~215 MB, one-off)")
                _download(IMAGES_ZIP, zip_path)
            _extract_images(zip_path, s.images_dir)

    seed_working_files(force=force)


def _extract_images(zip_path: Path, images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.lower().endswith(".png"):
                with zf.open(member) as src, open(images_dir / Path(member).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    log.info("✓ extracted %d images to %s", len(list(images_dir.glob("*.png"))), images_dir)


def seed_working_files(force: bool = False) -> None:
    """Create the canonical working files from the raw downloads (never overwrites edits unless forced)."""
    s = get_settings()

    if force or not s.restaurants_path.exists():
        raw = json.loads((s.raw_dir / "structured-restaurant-data.json").read_text(encoding="utf-8"))
        restaurants = []
        for i, rec in enumerate(raw):
            r = Restaurant.model_validate(rec)
            r.item_id = FIRST_ITEM_ID + i
            restaurants.append(r.model_dump())
        s.restaurants_path.write_text(json.dumps(restaurants, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("✓ seeded %s (%d restaurants)", s.restaurants_path.name, len(restaurants))

    if force or not s.reviews_path.exists():
        shutil.copy(s.raw_dir / "augmented-user-review.json", s.reviews_path)
        log.info("✓ seeded %s", s.reviews_path.name)

    if force or not s.recipes_path.exists():
        shutil.copy(s.raw_dir / "Recipes.json", s.recipes_path)
        log.info("✓ seeded %s", s.recipes_path.name)
