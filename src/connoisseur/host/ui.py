# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/host/ui.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Gradio front-end for the MCP host.

Tabs
----
* **Concierge Chat** - ReAct agent that picks MCP tools on its own; the tool
  trace is shown inline as collapsible bubbles.
* **Personalized Picks** - calls the multi-agent ``personalized_recommendations``
  MCP tool directly with a user's history / preferences.
* **Knowledge Base** - admin view: browse, add (LLM extraction + validation)
  and delete restaurants. Write operations are deliberately *not* exposed as
  MCP tools, so the chat agent can never modify data.
* **About** - architecture overview.
"""

from __future__ import annotations

import asyncio
import json
import logging

import gradio as gr

from ..config import get_settings, setup_logging
from ..knowledge_base import KnowledgeBase
from ..llm import LLMConfigurationError
from ..retrieval.vector_store import RESTAURANTS, VectorStore
from .agent import ConnoisseurAgent

log = logging.getLogger(__name__)

agent = ConnoisseurAgent()
kb = KnowledgeBase()

QUICK_PROMPTS = [
    ("Find moody restaurants", "Find me some moody restaurants"),
    ("Tell me about Iron & Embers", "Tell me about Iron & Embers"),
    ("Zen dining in Little Tokyo?", "What's a zen dining experience in Little Tokyo?"),
    ("Romantic dinner under $$$", "Suggest a romantic dinner spot that costs $$$ or less"),
    ("Spicy noodles to cook", "I want to cook spicy noodles at home tonight - any recipes?"),
]

EXAMPLE_PROFILES = [
    """Restaurant visit history:
- "Green Bowl" (Vegan, $$) 8 times
- "Mediterranean Grill" (Mediterranean, $$) 5 times
Posts: "Loving my plant-based journey! 🌱", "This gluten-free Mediterranean bowl is amazing!"
Dietary restrictions: vegan, gluten-free. Budget: moderate.""",
    """Restaurant visit history:
- "Omakase Sushi" (Japanese fine dining, $$$$) 4 times
- "Street Food Market" (International fusion, $$) 6 times
Posts: "Mind-blown by the 12-course tasting menu! 🤯", "This molecular take on ramen is art."
Dietary restrictions: none. I love trying bold new things.""",
]


def _friendly_error(e: Exception) -> str:
    if isinstance(e, LLMConfigurationError) or "LLMConfigurationError" in repr(e):
        return f"⚙️ **LLM not configured.** {e}\n\nCopy `.env.example` to `.env`, add your credentials and restart."
    return f"⚠️ Something went wrong: `{type(e).__name__}: {e}`"


# ---------------------------------------------------------------- chat tab
async def handle_chat(user_message: str, history: list | None):
    history = list(history or [])
    if not user_message or not user_message.strip():
        yield history
        return
    prior = list(history)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": "Thinking…"})
    yield history

    history.pop()  # drop the placeholder once real events arrive
    try:
        async for event in agent.stream(user_message, prior):
            if event.kind == "tool_call":
                history.append(
                    {
                        "role": "assistant",
                        "content": f"```json\n{event.content}\n```",
                        "metadata": {"title": f"🛠️ Calling `{event.tool}`", "status": "pending"},
                    }
                )
            elif event.kind == "tool_result":
                history[-1]["metadata"]["status"] = "done"
                history[-1]["content"] += f"\n**Result (truncated):**\n```\n{event.content}\n```"
            elif event.kind == "final":
                history.append({"role": "assistant", "content": event.content})
            yield history
    except LLMConfigurationError as e:
        log.warning("chat unavailable: %s", e)
        history.append({"role": "assistant", "content": _friendly_error(e)})
        yield history
    except Exception as e:
        log.exception("chat failed")
        history.append({"role": "assistant", "content": _friendly_error(e)})
        yield history


# ------------------------------------------------------ personalized tab
async def handle_personalized(user_context: str, top_n: int):
    if not user_context.strip():
        yield "Please describe your tastes, history or dietary needs first."
        return
    yield "⏳ Six agents at work: profiling → retrieving → analysing trends, styles & nutrition in parallel → synthesising…"
    try:
        async with agent.client as client:
            result = await client.call_tool(
                "personalized_recommendations", {"user_context": user_context, "top_n": int(top_n)}
            )
        payload = result.structured_content or json.loads(result.content[0].text)
        md = payload.get("markdown", "")
        if payload.get("errors"):
            md += (
                "\n\n<details><summary>Agent warnings</summary>\n\n"
                + "\n".join(f"- {e}" for e in payload["errors"])
                + "\n</details>"
            )
        yield md
    except Exception as e:
        log.exception("personalized recommendations failed")
        yield _friendly_error(e)


# ---------------------------------------------------- knowledge-base tab
KB_COLUMNS = ["name", "neighborhood", "cuisine", "type", "rating", "price_range", "vibes"]


def kb_table():
    return [
        [r.name, r.neighborhood, r.cuisine, r.type, r.rating, r.price_range, ", ".join(r.vibes)]
        for r in kb.restaurants()
    ]


def kb_status() -> str:
    counts = VectorStore().counts()
    return (
        f"**{len(kb.restaurants())}** restaurants · **{len(kb.recipes())}** recipes · "
        f"vector index → restaurants: {counts['restaurants']}, recipes: {counts['recipes']}, "
        f"dish images: {counts['recipe_images']}"
    )


def _sync_index(add=None, remove=None) -> str:
    store = VectorStore()
    if not store.has(RESTAURANTS):
        return " (vector index not built - keyword search will be used)"
    if add is not None:
        store.upsert_restaurants([add])
    if remove is not None:
        store.delete_restaurant(remove)
    return " and the vector index was updated"


async def kb_add(description: str, confirmed: bool):
    if not confirmed:
        return (
            "❗ Tick the confirmation box - changes are written to the database immediately.",
            kb_table(),
            kb_status(),
        )
    if not description.strip():
        return "Paste a restaurant description first.", kb_table(), kb_status()
    try:
        from ..ingestion.extractor import RestaurantExtractor

        restaurant = await asyncio.to_thread(RestaurantExtractor().extract, description)
        if restaurant is None:
            return "❌ Could not structure that description (validation failed after repairs).", kb_table(), kb_status()
        kb.add_restaurant(restaurant)
        note = await asyncio.to_thread(_sync_index, restaurant)
        msg = f"✅ Added **{restaurant.name}**{note}.\n```json\n{restaurant.model_dump_json(indent=2)}\n```"
    except Exception as e:
        msg = _friendly_error(e)
    return msg, kb_table(), kb_status()


async def kb_delete(name: str, confirmed: bool):
    if not confirmed:
        return "❗ Tick the confirmation box to delete.", kb_table(), kb_status()
    try:
        removed = kb.delete_restaurant(name.strip())
        note = await asyncio.to_thread(_sync_index, None, removed)
        msg = f"🗑️ Deleted **{removed.name}**{note}. A backup was kept as `restaurants.json.bak`."
    except KeyError as e:
        msg = f"Not found: {e}"
    return msg, kb_table(), kb_status()


ABOUT_MD = """
## How it works

```
 Gradio UI ──► MCP Host (ReAct agent, LangChain) ──MCP──► Connoisseur MCP Server
                    │  sampling ◄──────────────────────────┤  tools · resources · prompts
                    ▼                                       ├─ Knowledge base (JSON)
               LLM (watsonx / OpenAI / Anthropic)           ├─ ChromaDB: MiniLM text + CLIP image index
                                                            └─ LangGraph 6-agent workflow
```

* **Module 1 - Data:** LLM extraction of free-text restaurant descriptions into validated JSON (Pydantic + auto-repair loop); vision-LLM captions for dish photos.
* **Module 2 - Multimodal RAG:** MiniLM text and CLIP image embeddings in ChromaDB, metadata-filtered similarity search and min-max-normalised score fusion.
* **Module 3 - Agents:** six specialised agents in a LangGraph workflow with a parallel analysis phase and a grounding guardrail.
* **Module 4 - MCP:** everything is exposed through an MCP server, so any MCP-capable host can use it.

Designed and developed by **Gautam Kakadiya** as the capstone of the **IBM RAG and Agentic AI Professional Certificate**.

© 2026 Gautam Kakadiya · MIT License
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Connoisseur Companion") as demo:
        gr.Markdown(
            "# 🍷 Connoisseur Companion\n"
            "Your AI guide to California's restaurant scene - ask by name, cuisine, vibe or craving."
        )
        with gr.Tabs():
            with gr.Tab("💬 Concierge Chat"):
                chatbot = gr.Chatbot(height=520, label="Connoisseur Companion")
                msg = gr.Textbox(
                    label="Ask about restaurants or recipes",
                    placeholder='e.g. "Find me a moody spot in DTLA" or "Tell me about Sakura Garden"',
                )
                with gr.Row():
                    buttons = [(gr.Button(label, size="sm"), prompt) for label, prompt in QUICK_PROMPTS]
                clear = gr.Button("🧹 Clear conversation", size="sm", variant="secondary")

                msg.submit(handle_chat, [msg, chatbot], [chatbot])
                msg.submit(lambda: "", None, msg)
                for button, prompt in buttons:
                    button.click(handle_chat, [gr.State(prompt), chatbot], [chatbot])
                clear.click(lambda: [], None, chatbot)

            with gr.Tab("🎯 Personalized Picks"):
                gr.Markdown(
                    "Describe your dining history, favourite posts, dietary needs and budget. "
                    "Six specialised agents will profile you, retrieve candidates and recommend restaurants **and** recipes."
                )
                context = gr.Textbox(label="About you", lines=8, placeholder=EXAMPLE_PROFILES[0])
                gr.Examples(EXAMPLE_PROFILES, inputs=context, label="Example personas")
                top_n = gr.Slider(1, 5, value=3, step=1, label="Recommendations per category")
                run = gr.Button("Get my recommendations", variant="primary")
                output = gr.Markdown()
                run.click(handle_personalized, [context, top_n], output)

            with gr.Tab("🗂️ Knowledge Base"):
                status = gr.Markdown(kb_status)
                table = gr.Dataframe(value=kb_table, headers=KB_COLUMNS, interactive=False, wrap=True)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ➕ Add a restaurant from a free-text description")
                        desc = gr.Textbox(
                            lines=5,
                            label="Description",
                            placeholder="The Copper Sprout is a Modern Appalachian farm-to-table spot in Pasadena…",
                        )
                        add_ok = gr.Checkbox(label="I understand this writes to the database")
                        add_btn = gr.Button("Extract & add", variant="primary")
                    with gr.Column():
                        gr.Markdown("### 🗑️ Delete a restaurant")
                        del_name = gr.Textbox(label="Exact restaurant name")
                        del_ok = gr.Checkbox(label="I understand this deletes the record")
                        del_btn = gr.Button("Delete", variant="stop")
                kb_msg = gr.Markdown()
                add_btn.click(kb_add, [desc, add_ok], [kb_msg, table, status])
                del_btn.click(kb_delete, [del_name, del_ok], [kb_msg, table, status])

            with gr.Tab("ℹ️ About"):
                gr.Markdown(ABOUT_MD)
    return demo


def main(share: bool = False, host: str | None = None, port: int | None = None) -> None:
    setup_logging()
    s = get_settings()
    build_ui().queue().launch(
        share=share,
        server_name=host or s.ui_host,
        server_port=port or s.ui_port,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
