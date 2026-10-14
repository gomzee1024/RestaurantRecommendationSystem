# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/cli.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""``connoisseur`` command-line interface - one entry point for every pipeline stage.

connoisseur download-data [--with-images]   fetch datasets into ./data
connoisseur ingest [--limit N]              LLM-extract all restaurants from the culinary map
connoisseur caption-recipes [--limit N]     vision-LLM captions for dish photos
connoisseur build-index [--no-images]       build the ChromaDB multimodal index
connoisseur status                          knowledge-base + index summary
connoisseur kb                              interactive knowledge-base manager
connoisseur recommend "I love spicy..."     run the multi-agent workflow from the terminal
connoisseur serve [--transport http]        run the MCP server
connoisseur check [--sampling]              verify the MCP server end-to-end
connoisseur ui [--share]                    launch the Gradio app
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import get_settings, setup_logging
from .knowledge_base import KnowledgeBase

log = logging.getLogger("connoisseur")


# ------------------------------------------------------------------ commands
def cmd_download(args) -> None:
    from .ingestion.download import download_all

    download_all(with_images=args.with_images, force=args.force)
    print(f"✅ Data ready in {get_settings().data_dir}")


def cmd_ingest(args) -> None:
    """Grow the knowledge base from the 20 seed records to every restaurant in the culinary map."""
    from .ingestion.extractor import RestaurantExtractor, split_culinary_map

    kb = KnowledgeBase()
    known = {r.name.lower() for r in kb.restaurants()}
    paragraphs = [p for p in split_culinary_map(kb.culinary_map()) if not any(n in p.lower() for n in known)]
    if args.limit:
        paragraphs = paragraphs[: args.limit]
    print(f"Extracting {len(paragraphs)} new restaurant descriptions ({len(known)} already in the knowledge base)…")

    extractor = RestaurantExtractor()
    added = skipped = 0
    for i, paragraph in enumerate(paragraphs, 1):
        restaurant = extractor.extract(paragraph)
        if restaurant is None:
            skipped += 1
        else:
            try:
                kb.add_restaurant(restaurant)  # saved immediately -> the run is resumable
                added += 1
            except ValueError:
                skipped += 1  # duplicate name
        if i % 10 == 0 or i == len(paragraphs):
            print(f"  {i}/{len(paragraphs)} processed · {added} added · {skipped} skipped")
    print(f"✅ Done. Knowledge base now holds {len(kb.restaurants())} restaurants. Run `connoisseur build-index` next.")


def cmd_caption(args) -> None:
    from .ingestion.captioner import ImageCaptioner

    s, kb = get_settings(), KnowledgeBase()
    recipes = kb.recipes()
    todo = [r for r in recipes if not r.image_description and (s.images_dir / r.image_filename).exists()]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("Nothing to caption (already captioned, or run `connoisseur download-data --with-images`).")
        return
    captioner = ImageCaptioner()
    for i, recipe in enumerate(todo, 1):
        try:
            recipe.image_description = captioner.caption_dish(s.images_dir / recipe.image_filename, recipe.name)
        except Exception as e:
            log.warning("caption failed for %s: %s", recipe.name, e)
        if i % 10 == 0 or i == len(todo):
            kb.save_recipes(recipes)  # checkpoint
            print(f"  {i}/{len(todo)} captioned")
    print("✅ Captions saved. Re-run `connoisseur build-index` so recipe search uses them.")


def cmd_build_index(args) -> None:
    from .retrieval.vector_store import build_index

    print("Building vector index (first run downloads the MiniLM and CLIP models)…")
    counts = build_index(KnowledgeBase(), include_images=not args.no_images)
    for name, n in counts.items():
        print(f"  ✓ {name:<14} {n} vectors")
    print("🎉 Multimodal vector index construction complete")


def cmd_status(_args) -> None:
    from .retrieval.service import RetrievalService

    s = get_settings()
    print(
        json.dumps(
            {"llm_provider": s.llm_provider, "data_dir": str(s.data_dir), **RetrievalService().status()}, indent=2
        )
    )


def cmd_recommend(args) -> None:
    from .agents.workflow import RecommendationWorkflow, format_recommendations

    state = RecommendationWorkflow().run(args.text, top_n=args.top_n)
    print(format_recommendations(state))
    if state.get("errors"):
        print("\nWarnings:", *state["errors"], sep="\n - ")


def cmd_serve(args) -> None:
    from .mcp_server import main

    main(transport=args.transport, host=args.host, port=args.port)


def cmd_check(args) -> None:
    from .mcp_client import run_checks

    sys.exit(0 if asyncio.run(run_checks(with_sampling=args.sampling)) else 1)


def cmd_ui(args) -> None:
    from .host.ui import main

    main(share=args.share, host=args.host, port=args.port)


def cmd_kb(_args) -> None:
    """Interactive manager: browse / view / add / edit / delete, with a write-mode confirmation."""
    kb = KnowledgeBase()
    while True:
        items = kb.restaurants()
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(items)}")
        print("1. Browse all  2. View record  3. Add (LLM extraction)  4. Edit  5. Delete  6. Exit")
        choice = input("Action: ").strip()
        if choice == "6":
            return
        if choice == "1":
            for i, r in enumerate(items):
                print(f"{i:>3}. {r.name}")
            continue
        if choice not in {"2", "3", "4", "5"}:
            print("Invalid input.")
            continue

        if choice in {"3", "4", "5"}:
            print("\n❗ SECURITY WARNING: write mode - changes are saved immediately (a .bak backup is kept).")
            if input("Type 'yes' to proceed: ").strip().lower() != "yes":
                print("Operation cancelled.")
                continue

        if choice == "3":
            from .ingestion.extractor import RestaurantExtractor

            restaurant = RestaurantExtractor().extract(input("Restaurant description: "))
            if restaurant is None:
                print("❌ Could not structure that description. Nothing was added.")
                continue
            try:
                kb.add_restaurant(restaurant)
                print(f"✅ Added {restaurant.name}. Run `connoisseur build-index` to make it searchable.")
            except ValueError as e:
                print(f"❌ {e}")
            continue

        try:
            index = int(input("Record index: "))
            record = items[index]
        except (ValueError, IndexError):
            print("Invalid index.")
            continue
        if choice == "2":
            print(json.dumps(record.model_dump(), indent=2, ensure_ascii=False))
        elif choice == "4":
            changes = {}
            for key, value in record.model_dump(exclude={"item_id"}).items():
                new = input(f"{key} [{value}] (Enter to skip): ").strip()
                if new:
                    try:
                        changes[key] = json.loads(new)  # numbers and lists, e.g. 4.5 or ["a", "b"]
                    except json.JSONDecodeError:
                        changes[key] = new
            if changes:
                kb.update_restaurant(record.name, **changes)
                print("✅ Record updated.")
        elif choice == "5":
            print(f"✅ Deleted: {kb.delete_restaurant(record.name).name}")


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="connoisseur", description="Connoisseur Companion - RAG + agentic restaurant recommender"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download-data", help="fetch datasets into ./data")
    d.add_argument("--with-images", action="store_true", help="also fetch the ~215 MB recipe image archive")
    d.add_argument("--force", action="store_true", help="re-download and re-seed working files")
    d.set_defaults(func=cmd_download)

    i = sub.add_parser("ingest", help="LLM-extract every restaurant from the culinary map")
    i.add_argument("--limit", type=int)
    i.set_defaults(func=cmd_ingest)

    c = sub.add_parser("caption-recipes", help="vision-LLM captions for recipe photos")
    c.add_argument("--limit", type=int)
    c.set_defaults(func=cmd_caption)

    b = sub.add_parser("build-index", help="build the ChromaDB multimodal vector index")
    b.add_argument("--no-images", action="store_true", help="skip the CLIP image collection")
    b.set_defaults(func=cmd_build_index)

    sub.add_parser("status", help="knowledge-base and index summary").set_defaults(func=cmd_status)
    sub.add_parser("kb", help="interactive knowledge-base manager").set_defaults(func=cmd_kb)

    r = sub.add_parser("recommend", help="run the multi-agent workflow")
    r.add_argument("text", help="your tastes, history, dietary needs…")
    r.add_argument("--top-n", type=int, default=3)
    r.set_defaults(func=cmd_recommend)

    s = sub.add_parser("serve", help="run the MCP server")
    s.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(func=cmd_serve)

    k = sub.add_parser("check", help="verify the MCP server end-to-end")
    k.add_argument("--sampling", action="store_true", help="also test MCP sampling (needs LLM credentials)")
    k.set_defaults(func=cmd_check)

    u = sub.add_parser("ui", help="launch the Gradio app")
    u.add_argument("--share", action="store_true", help="create a public gradio.live link")
    u.add_argument("--host")
    u.add_argument("--port", type=int)
    u.set_defaults(func=cmd_ui)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252 and choke on emoji
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
