# =============================================================================
#  Connoisseur Companion - Restaurant Recommendation System
#  File   : src/connoisseur/mcp_connection.py
#  Author : Gautam Kakadiya
#  Copyright (c) 2026 Gautam Kakadiya
#  Licensed under the MIT License. See LICENSE in the project root.
# =============================================================================

"""Factory for MCP client connections used by the host app and the diagnostic client.

* **Transport** - spawns the server as a stdio subprocess by default, or talks
  to a remote server over Streamable HTTP when ``MCP_SERVER_URL`` is set.
* **Roots** - declares the project directory as the only filesystem root the
  server may use.
* **Sampling** - fulfils the server's LLM requests with the *client's* model,
  so API keys stay on the client side.
"""

from __future__ import annotations

import logging
import os
import sys

from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from langchain_core.messages import HumanMessage, SystemMessage
from mcp.types import Root

from .config import PROJECT_ROOT, get_settings
from .llm import get_chat_model, message_text

log = logging.getLogger(__name__)


def project_roots(_context=None) -> list[Root]:
    return [Root(uri=PROJECT_ROOT.as_uri(), name=PROJECT_ROOT.name)]


async def llm_sampling_handler(messages, params, _context) -> str:
    """Run an LLM call on behalf of the server (MCP sampling)."""
    lc_messages = []
    if getattr(params, "systemPrompt", None):
        lc_messages.append(SystemMessage(content=params.systemPrompt))
    for m in messages:
        text = getattr(m.content, "text", None) or str(m.content)
        lc_messages.append(HumanMessage(content=text) if m.role == "user" else SystemMessage(content=text))
    log.info("[sampling] server requested an LLM completion (%d message(s))", len(messages))
    model = get_chat_model(max_tokens=getattr(params, "maxTokens", None) or 300)
    reply = await model.ainvoke(lc_messages)
    return message_text(reply.content)


def make_transport(keep_alive: bool = True):
    s = get_settings()
    if s.mcp_server_url:
        return StreamableHttpTransport(s.mcp_server_url)
    env = dict(os.environ)  # stdio servers don't inherit the parent env by default
    src = PROJECT_ROOT / "src"
    if (src / "connoisseur").exists():  # running from a checkout without `pip install`
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(src), env.get("PYTHONPATH")]))
    return StdioTransport(
        command=sys.executable,
        args=["-m", "connoisseur.mcp_server"],
        env=env,
        cwd=str(PROJECT_ROOT),
        # True for long-lived hosts (one warm server reused across chat turns);
        # False for one-shot commands so the subprocess exits with the client.
        keep_alive=keep_alive,
    )


def make_client(with_sampling: bool = True, keep_alive: bool = True) -> Client:
    return Client(
        make_transport(keep_alive),
        roots=project_roots,
        sampling_handler=llm_sampling_handler if with_sampling else None,
        timeout=600,  # the multi-agent tool can take a while
    )
