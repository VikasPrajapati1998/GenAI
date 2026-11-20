"""
AgenticRAG.py — Agentic RAG with native tool calling (ReAct-style loop)
======================================================================

    ┌──────────────── agent loop (max MAX_STEPS) ─────────────────┐
    │  LLM sees: system + question + tool results so far           │
    │  LLM decides: call tool(s)  ──► code executes ──► results ───┘
    │           or: final answer  ──► exit                         │
    └──────────────────────────────────────────────────────────────┘

Instead of a fixed retrieve-then-generate pipeline, the model chooses which source to query, in what
order, and when it has enough evidence (section 4.5 / 8.2 of the guide). Tools here:
    search_policies(query)        semantic search over policy text (vector RAG as a tool)
    query_database(sql)           read-only SQL over quotes/vendors (SQLite, validated)
    check_vendor(vendor_name)     vendor master lookup
    calculate(expression)         safe arithmetic (LLMs are bad at maths; tools aren't)

Ollama's Python SDK accepts plain Python functions as tools: it builds the JSON schema from type hints
and the Google-style docstring.

Run
---
    ollama pull granite3.3:2b && ollama pull nomic-embed-text:v1.5
    python AgenticRAG.py
    python AgenticRAG.py --model qwen3:4b        # better tool use (ollama pull qwen3:4b)
    python AgenticRAG.py --model gpt-oss:120b-cloud
Models must support tools in Ollama (granite3.3, llama3.2, qwen3, gpt-oss do; gemma3 does not).
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import sqlite3
import sys

import numpy as np
import ollama

# Windows consoles: make sure arrows/box characters in the logs print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHAT_MODEL = os.getenv("CHAT_MODEL", "granite3.3:2b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:v1.5")
MAX_STEPS = 6

POLICIES = [
    "Purchases up to INR 2,00,000 need department head approval.",
    "Purchases from INR 2,00,001 to INR 10,00,000 need Finance Controller approval.",
    "Purchases above INR 10,00,000 need CFO approval and three competitive quotes.",
    "Only APPROVED vendors with valid GST and an unexpired ISO 9001 certificate may receive hardware POs.",
    "Every GPU purchase needs an IT Security review before the PO is raised.",
]

# ── a tiny database ──────────────────────────────────────────────────────────
DB = sqlite3.connect(":memory:", check_same_thread=False)
DB.executescript("""
CREATE TABLE vendors(name TEXT PRIMARY KEY, status TEXT, iso_valid_until TEXT, gst INTEGER);
CREATE TABLE quotes(vendor TEXT, product TEXT, unit_price_inr INTEGER, valid_until TEXT);
INSERT INTO vendors VALUES ('ABC Technologies','APPROVED','2027-03-31',1),('Zenith Systems','APPROVED','2026-10-31',1),
                           ('Nova Compute','BLACKLISTED',NULL,1),('Orbit Infotech','APPROVED','2028-01-15',1);
INSERT INTO quotes VALUES ('ABC Technologies','RTX 6000 Ada',650000,'2026-12-31'),('Zenith Systems','RTX 6000 Ada',620000,'2026-12-15'),
                          ('Nova Compute','RTX 6000 Ada',590000,'2026-12-31'),('Orbit Infotech','RTX 6000 Ada',668000,'2027-01-31');
""")


def _embed(texts, kind):
    if "nomic" in EMBED_MODEL:
        texts = [("search_document: " if kind == "document" else "search_query: ") + t for t in texts]
    v = np.array(ollama.embed(model=EMBED_MODEL, input=texts).embeddings, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


_POLICY_VECS = None


# ── tools (type hints + docstrings become the JSON schema the model sees) ──────
def search_policies(query: str) -> str:
    """Search company procurement policies by meaning.

    Args:
        query: what to look for, e.g. 'approval threshold' or 'vendor eligibility'

    Returns:
        The most relevant policy sentences.
    """
    global _POLICY_VECS
    if _POLICY_VECS is None:
        _POLICY_VECS = _embed(POLICIES, "document")
    s = _POLICY_VECS @ _embed([query], "query")[0]
    return "\n".join(POLICIES[i] for i in np.argsort(-s)[:3])


def query_database(sql: str) -> str:
    """Run a read-only SQL SELECT on the procurement database.

    Args:
        sql: a single SELECT statement. Tables: vendors(name, status, iso_valid_until, gst),
             quotes(vendor, product, unit_price_inr, valid_until)

    Returns:
        Rows as JSON, or an error message.
    """
    s = sql.strip().rstrip(";")
    if not s.lower().startswith("select") or ";" in s:
        return "ERROR: only a single SELECT statement is allowed"
    try:
        cur = DB.execute(s)
        cols = [c[0] for c in cur.description]
        return json.dumps([dict(zip(cols, r)) for r in cur.fetchmany(20)])
    except sqlite3.Error as e:
        return f"ERROR: {e}"


def check_vendor(vendor_name: str) -> str:
    """Look up a vendor's status and certificate validity in the vendor master.

    Args:
        vendor_name: vendor name, e.g. 'Zenith Systems'

    Returns:
        Vendor record as JSON, or 'not found'.
    """
    row = DB.execute("SELECT * FROM vendors WHERE lower(name) LIKE ?", (f"%{vendor_name.lower()}%",)).fetchone()
    if not row:
        return "not found"
    return json.dumps({"name": row[0], "status": row[1], "iso_valid_until": row[2], "gst": bool(row[3]),
                       "note": "today is 2026-11-20; an ISO date earlier than today means expired"})


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely.

    Args:
        expression: e.g. '620000 * 2'

    Returns:
        The numeric result.
    """
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("unsupported expression")
    try:
        return str(ev(ast.parse(expression.replace(",", ""), mode="eval").body))
    except Exception as e:  # noqa: BLE001 — return errors to the model so it can retry
        return f"ERROR: {e}"


TOOLS = {f.__name__: f for f in (search_policies, query_database, check_vendor, calculate)}

SYSTEM = (
    "You are a procurement agent. Use the tools to gather facts before answering; never guess prices, "
    "statuses or rules. Typical approach: query quotes, check each candidate vendor, look up the policy, "
    "calculate totals. When you have enough evidence, give a short final answer that cites the facts used."
)


def run_agent(question: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for step in range(1, MAX_STEPS + 1):
        kw = dict(model=CHAT_MODEL, messages=messages, tools=list(TOOLS.values()), options={"temperature": 0})
        try:
            resp = ollama.chat(think=False, **kw)
        except ollama.ResponseError as e:
            if "think" not in str(e).lower():
                raise
            resp = ollama.chat(**kw)
        msg = resp.message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content.strip()
        for call in msg.tool_calls:
            name, args = call.function.name, dict(call.function.arguments or {})
            fn = TOOLS.get(name)
            try:
                result = fn(**args) if fn else f"ERROR: unknown tool {name}"
            except TypeError as e:
                result = f"ERROR: bad arguments: {e}"
            print(f"  [step {step}] {name}({json.dumps(args)[:90]}) → {result[:110].replace(chr(10), ' ')}")
            messages.append({"role": "tool", "tool_name": name, "content": result})
    # Budget exhausted: ask for a best-effort answer without tools
    messages.append({"role": "user", "content": "Step budget reached. Answer now with what you have."})
    return ollama.chat(model=CHAT_MODEL, messages=messages).message.content.strip()


def main():
    global CHAT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--question")
    ap.add_argument("--chat", action="store_true")
    a = ap.parse_args()
    CHAT_MODEL = a.model
    qs = [a.question] if a.question else [
        "Which vendor can supply two RTX 6000 Ada GPUs at the lowest price, is it eligible, and who approves the total?"]
    try:
        if a.chat:
            while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
                print("Agent:", run_agent(q))
            return
        for q in qs:
            print(f"Q: {q}")
            print("A:", run_agent(q))
    except (ConnectionError, ollama.ResponseError) as e:
        sys.exit(f"Ollama error: {e}\n(Does the model support tools? Try --model qwen3:4b)")


if __name__ == "__main__":
    main()
