# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : app.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Hugging Face Spaces / one-file entry point.

Bootstraps the data and vector index on first start, then launches the Gradio app.
Locally, prefer `connoisseur ui`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))  # works without `pip install -e .`

from connoisseur.config import get_settings, setup_logging  # noqa: E402

setup_logging()
settings = get_settings()

if not settings.restaurants_path.exists():
    from connoisseur.ingestion.download import download_all

    download_all(with_images=True)

if not settings.chroma_dir.exists():
    from connoisseur.knowledge_base import KnowledgeBase
    from connoisseur.retrieval.vector_store import build_index

    build_index(KnowledgeBase())

from connoisseur.host.ui import main  # noqa: E402

if __name__ == "__main__":
    main(host="0.0.0.0")
