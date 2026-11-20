"""
MCPAgent.py — Tool-using agent whose tools come from an MCP server
=================================================================

    ┌────────────── MCPAgent.py (host + MCP client) ──────────────┐        stdio        ┌──────────────────────────┐
    │ 1. connect & list_tools()  ──────────────────────────────────┼────────────────────►│ mcp_procurement_server.py │
    │ 2. convert MCP tool schemas → Ollama tool definitions       │                     │  get_quotes · check_vendor │
    │ 3. agent loop: LLM picks tool ─► call_tool() ─► result ─► LLM│◄────────────────────┤  approval_level · …        │
    └─────────────────────────────────────────────────────────────┘                     └──────────────────────────┘

Why MCP (section 9 of the guide): the agent does not hard-code tools. Any MCP server (ERP, SQL, GitHub,
filesystem …) can be plugged in, and the same server works with Claude, ChatGPT, IDEs and other hosts.
This file also shows two safety habits: an allow-list of tools the agent may call, and a write action
that the server blocks unless a human confirmed it.

Run
---
    pip install "mcp>=2" ollama
    ollama pull granite3.3:2b          # any tool-capable model: qwen3:4b, llama3.2, gpt-oss:120b-cloud
    python MCPAgent.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import ollama
from mcp import Client, StdioServerParameters

# Windows consoles: make sure arrows/box characters in the logs print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHAT_MODEL = os.getenv("CHAT_MODEL", "granite3.3:2b")
SERVER = Path(__file__).with_name("mcp_procurement_server.py")
MAX_STEPS = 6
ALLOWED_TOOLS = {"get_quotes", "check_vendor", "approval_level", "raise_purchase_request"}   # tool allow-list


def mcp_to_ollama(tool) -> dict:
    """MCP Tool (name, description, input_schema) → Ollama/OpenAI-style function tool."""
    return {"type": "function", "function": {
        "name": tool.name, "description": tool.description or "",
        "parameters": tool.input_schema or {"type": "object", "properties": {}}}}


def result_text(res) -> str:
    """Prefer structured output; fall back to text content blocks."""
    if res.structured_content:
        return json.dumps(res.structured_content, separators=(',', ':'))
    return "\n".join(getattr(c, "text", str(c)) for c in res.content)


def llm(messages, tools):
    kw = dict(model=CHAT_MODEL, messages=messages, tools=tools, options={"temperature": 0})
    try:
        return ollama.chat(think=False, **kw).message
    except ollama.ResponseError as e:
        if "think" not in str(e).lower():
            raise
        return ollama.chat(**kw).message


async def run(question: str) -> str:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with Client(params) as client:
        tools = [t for t in (await client.list_tools()).tools if t.name in ALLOWED_TOOLS]
        print(f"[mcp] connected to '{SERVER.name}', tools: {[t.name for t in tools]}")
        policy = await client.read_resource("policy://procurement")       # resources = read-only context
        policy_text = "\n".join(getattr(c, "text", "") for c in policy.contents)
        ollama_tools = [mcp_to_ollama(t) for t in tools]

        messages = [
            {"role": "system", "content": "You are a procurement agent. Use the tools for every fact; do not "
                                          "create purchase requests unless the user explicitly confirmed.\n"
                                          f"Policy:\n{policy_text}"},
            {"role": "user", "content": question}]
        for step in range(1, MAX_STEPS + 1):
            msg = llm(messages, ollama_tools)
            messages.append(msg)
            if not msg.tool_calls:
                return msg.content.strip()
            for call in msg.tool_calls:
                name, args = call.function.name, dict(call.function.arguments or {})
                if name not in ALLOWED_TOOLS:
                    out = f"ERROR: tool '{name}' is not allowed"
                else:
                    try:
                        out = result_text(await client.call_tool(name, args))
                    except Exception as e:  # noqa: BLE001 — surface tool errors to the model
                        out = f"ERROR: {e}"
                print(f"  [step {step}] {name}({json.dumps(args)[:80]}) → {out[:120].replace(chr(10), ' ')}")
                messages.append({"role": "tool", "tool_name": name, "content": out})
        return "Step budget reached without a final answer."


def main():
    global CHAT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--question", default="Find the cheapest RTX 6000 Ada quote from an eligible vendor for 2 units "
                                          "and tell me who must approve it. Do not create a request yet.")
    a = ap.parse_args()
    CHAT_MODEL = a.model
    try:
        print("Q:", a.question)
        print("A:", asyncio.run(run(a.question)))
    except (ConnectionError, ollama.ResponseError) as e:
        sys.exit(f"Ollama error: {e}")


if __name__ == "__main__":
    main()
