"""
CAG.py — Cache-Augmented Generation (preload knowledge, reuse the KV cache)
==========================================================================

    PRELOAD (once):  [ system rules + ENTIRE knowledge base ]  ─► model processes it ─► KV cache kept in memory
    EACH QUERY    :  [ same cached prefix ] + [ new question ] ─► only the new tokens are processed ─► answer

Key idea (paper: "Don't Do RAG", arXiv 2412.15605): when the knowledge is small and stable, skip retrieval.
Put everything in the context once, keep the transformer's key/value cache, and answer every question
against that cached prefix. No vector DB, no retrieval errors; latency after warm-up is low.

How this maps to Ollama
-----------------------
Ollama (llama.cpp underneath) keeps the KV cache of the last prompt in each loaded model slot. If the next
request starts with the *exact same token prefix*, it reuses those cached tokens and only evaluates the new
suffix. So "CAG on Ollama" =
    1. a byte-identical static prefix (system prompt + documents) at the START of every request,
    2. `keep_alive` so the model (and its cache) stays loaded,
    3. a big enough `num_ctx` to hold the whole knowledge base,
    4. single-turn requests (we do NOT append history, which keeps the prefix identical).
Watch `prompt_eval_count` in the output: the first call evaluates the whole corpus; later calls evaluate
only the question tokens. (Exact numbers depend on your Ollama version and settings.)

Hosted-API equivalent: prompt caching (Anthropic cache_control, OpenAI automatic caching, Gemini caches).
Pure KV-cache CAG (as in the paper) is done with HF transformers by saving `past_key_values` and
truncating back to the prefix length after each answer.

Run
---
    ollama pull granite3.3:2b
    python CAG.py
    python CAG.py --chat
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import ollama

# Windows consoles: make sure arrows/box characters in the logs print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHAT_MODEL = os.getenv("CHAT_MODEL", "granite3.3:2b")
NUM_CTX = 8192          # must exceed the size of the preloaded knowledge + question + answer
KEEP_ALIVE = "30m"      # keep the model and its KV cache resident between questions

# Small, STABLE knowledge base: exactly the case CAG is designed for.
KNOWLEDGE = """
PROCUREMENT POLICY v3.2
1. Up to INR 2,00,000: department head approval.
2. INR 2,00,001 – 10,00,000: Finance Controller approval.
3. Above INR 10,00,000: CFO approval plus three competitive quotes.
4. Only vendors with status APPROVED may receive purchase orders; GST and ISO 9001 are mandatory for hardware.
5. Blacklisted vendors are barred for 24 months.
6. Emergency purchases may skip quotes but need CFO sign-off within 48 hours.

IT HARDWARE STANDARD 2026
- GPU workstations: NVIDIA RTX PRO 6000 Blackwell or RTX 6000 Ada.
- Every GPU purchase needs an IT Security review before the PO.
- Minimum warranty: 3 years on-site for servers and workstations.
- Developer laptops: 32 GB RAM minimum, refreshed every 4 years.

TRAVEL & EXPENSES
- Domestic flights: economy class. International flights over 8 hours: premium economy allowed.
- Hotel cap: INR 9,000 per night in metro cities, INR 6,000 elsewhere.
- Expense claims must be filed within 30 days with receipts.
""".strip()

# The static prefix. Must be IDENTICAL on every call for cache reuse — no timestamps, no user ids here.
SYSTEM_PREFIX = (
    "You are the company policy assistant. Answer strictly from the KNOWLEDGE below. "
    "Quote the rule number or section you used. If the knowledge does not cover the question, say so.\n\n"
    f"KNOWLEDGE:\n{KNOWLEDGE}"
)


def ask(question: str) -> tuple[str, dict]:
    """One CAG query: cached prefix + new question. Returns answer and timing stats."""
    messages = [{"role": "system", "content": SYSTEM_PREFIX},
                {"role": "user", "content": question}]
    kw = dict(model=CHAT_MODEL, messages=messages, keep_alive=KEEP_ALIVE,
              options={"num_ctx": NUM_CTX, "temperature": 0})
    t0 = time.perf_counter()
    try:
        r = ollama.chat(think=False, **kw)
    except ollama.ResponseError as e:
        if "think" not in str(e).lower():
            raise
        r = ollama.chat(**kw)
    stats = {
        "wall_s": time.perf_counter() - t0,
        "prompt_tokens_evaluated": r.prompt_eval_count,             # small after warm-up = cache hit
        "prompt_eval_ms": (r.prompt_eval_duration or 0) / 1e6,
        "output_tokens": r.eval_count,
    }
    return r.message.content.strip(), stats


def preload() -> None:
    """The CAG 'preload' step: process the knowledge once so its KV cache is resident."""
    print(f"[preload] loading ~{len(SYSTEM_PREFIX) // 4} tokens of knowledge into {CHAT_MODEL} …")
    _, s = ask("Reply with OK.")
    print(f"[preload] prompt tokens evaluated: {s['prompt_tokens_evaluated']}  "
          f"({s['prompt_eval_ms']:.0f} ms, wall {s['wall_s']:.1f}s)\n")


def main():
    global CHAT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--chat", action="store_true")
    a = ap.parse_args()
    CHAT_MODEL = a.model
    try:
        preload()
    except (ConnectionError, ollama.ResponseError) as e:
        sys.exit(f"Ollama error: {e}\nIs Ollama running? Try `ollama pull {CHAT_MODEL}`.")

    def show(q):
        ans, s = ask(q)
        print(f"Q: {q}\nA: {ans}")
        print(f"   ↳ prompt tokens evaluated: {s['prompt_tokens_evaluated']} "
              f"({s['prompt_eval_ms']:.0f} ms) · wall {s['wall_s']:.1f}s\n")

    if a.chat:
        while (q := input("You: ").strip()).lower() not in {"exit", "quit"}:
            show(q)
        return
    for q in ["Who approves a purchase of INR 6,50,000?",
              "Can I fly premium economy to Singapore (5 hours)?",
              "What must happen before buying a GPU?",
              "What is the hotel cap in Pune?"]:
        show(q)
    print("Note: compare 'prompt tokens evaluated' with the preload number. Low values mean the KV cache "
          "for the knowledge prefix was reused. Change one character in KNOWLEDGE and the cache is invalidated.")


if __name__ == "__main__":
    main()
