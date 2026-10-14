<!-- Connoisseur Companion | Author: Gautam Kakadiya | Copyright (c) 2026 Gautam Kakadiya | MIT License -->

# data/

Everything in this folder except this README is generated, and git ignores it.

```bash
connoisseur download-data --with-images   # raw datasets + working copies
connoisseur build-index                   # ChromaDB vector index -> data/chroma/
```

| Path | Created by | Contents |
|------|------------|----------|
| `raw/` | `download-data` | Original course files: culinary map text, seed restaurants, reviews, recipes, image zip |
| `restaurants.json` | `download-data`, `ingest`, UI/CLI edits | Canonical restaurant knowledge base (`restaurants.json.bak` = previous version) |
| `reviews.json` | `download-data` | User reviews with AI captions of the reviewers' photos |
| `recipes.json` | `download-data`, `caption-recipes` | Recipes (+ vision-LLM photo captions) |
| `recipe_images/` | `download-data --with-images` | 109 dish photos (`recipe{id}.png`) |
| `chroma/` | `build-index` | Persistent vector index (restaurants, recipes, recipe_images) |

The datasets come from the IBM Skills Network *RAG and Agentic AI* capstone.
