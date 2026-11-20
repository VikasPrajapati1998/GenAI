"""
MemoryChatbot.py — Chatbot with short-term + long-term memory (Mem0 / Letta style, simplified)
============================================================================================

    each turn:
      1. RECALL   embed user message ─► top-k long-term memories (similarity × recency)
      2. ANSWER   system + recalled memories + rolling summary + last N turns ─► LLM
      3. EXTRACT  LLM pulls durable facts from the exchange (JSON) ─► ADD / UPDATE (dedupe by similarity)
      4. COMPACT  when the short-term window is long, summarise old turns into a running summary

    long-term store = memory_store.json  (profile · preference · episodic facts, with timestamps)

What it shows (section 10 of the guide): memory is a separate system with explicit write, recall, update,
and forget policies, plus user controls. Commands in chat mode:
    /memories          list stored memories
    /forget <id>       delete one memory   (users must be able to inspect and delete)
    /reset             wipe the store
    /new               start a new "session" (short-term memory cleared, long-term kept)

Run
---
    ollama pull granite3.3:2b && ollama pull nomic-embed-text:v1.5
    python MemoryChatbot.py            # scripted demo across two sessions
    python MemoryChatbot.py --chat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import ollama

# Windows consoles: make sure arrows/box characters in the logs print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHAT_MODEL = os.getenv("CHAT_MODEL", "granite3.3:2b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:v1.5")
STORE = Path(__file__).with_name("memory_store.json")
RECALL_K = 4
DEDUPE_THRESHOLD = 0.88      # cosine above this = same memory → update instead of add
WINDOW = 6                   # short-term: keep last N messages verbatim
HALF_LIFE_DAYS = 30          # recency decay for ranking

EXTRACT_SCHEMA = {"type": "object", "properties": {"memories": {"type": "array", "items": {"type": "object",
    "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": ["profile", "preference", "episodic"]}},
    "required": ["text", "type"]}}}, "required": ["memories"]}


def chat(messages, **kw) -> str:
    try:
        r = ollama.chat(model=CHAT_MODEL, messages=messages, think=False, **kw)
    except ollama.ResponseError as e:
        if "think" not in str(e).lower():
            raise
        r = ollama.chat(model=CHAT_MODEL, messages=messages, **kw)
    return r.message.content.strip()


def embed(text: str, kind: str) -> list[float]:
    if "nomic" in EMBED_MODEL:
        text = ("search_document: " if kind == "document" else "search_query: ") + text
    v = np.array(ollama.embed(model=EMBED_MODEL, input=[text]).embeddings[0], dtype=np.float32)
    return (v / (np.linalg.norm(v) + 1e-9)).tolist()


class LongTermMemory:
    def __init__(self, path: Path = STORE):
        self.path = path
        self.items: list[dict] = json.loads(path.read_text()) if path.exists() else []

    def save(self):
        self.path.write_text(json.dumps(self.items, indent=1))

    def recall(self, query: str, k: int = RECALL_K) -> list[dict]:
        if not self.items:
            return []
        q = np.array(embed(query, "query"))
        now = time.time()
        scored = []
        for m in self.items:
            sim = float(np.dot(q, m["vec"]))
            age_days = (now - m["updated"]) / 86400
            recency = 0.5 ** (age_days / HALF_LIFE_DAYS)
            scored.append((0.85 * sim + 0.15 * recency, m))   # similarity dominates; recency breaks ties
        return [m for s, m in sorted(scored, key=lambda x: -x[0])[:k]]

    def upsert(self, text: str, mtype: str):
        vec = embed(text, "document")
        for m in self.items:
            if float(np.dot(vec, m["vec"])) > DEDUPE_THRESHOLD:
                print(f"   [memory] UPDATE {m['id']}: '{m['text']}' → '{text}'")
                m.update(text=text, type=mtype, vec=vec, updated=time.time())
                self.save()
                return
        mid = uuid.uuid4().hex[:6]
        self.items.append({"id": mid, "text": text, "type": mtype, "vec": vec,
                           "created": time.time(), "updated": time.time()})
        print(f"   [memory] ADD {mid} ({mtype}): {text}")
        self.save()

    def forget(self, mid: str) -> bool:
        before = len(self.items)
        self.items = [m for m in self.items if m["id"] != mid]
        self.save()
        return len(self.items) < before


class MemoryChatbot:
    def __init__(self):
        self.ltm = LongTermMemory()
        self.history: list[dict] = []     # short-term window
        self.summary = ""                 # compacted older turns

    def new_session(self):
        self.history, self.summary = [], ""

    def extract(self, user: str, bot: str):
        raw = chat([{"role": "user", "content":
            "From this exchange, extract durable facts about the USER worth remembering for future "
            "conversations (identity, role, preferences, projects, decisions, events). Ignore small talk "
            "and facts about the assistant. Return an empty list if nothing is worth saving.\n\n"
            f"User: {user}\nAssistant: {bot}"}], format=EXTRACT_SCHEMA, options={"temperature": 0})
        try:
            for m in json.loads(raw).get("memories", [])[:3]:
                if len(m.get("text", "")) > 8:
                    self.ltm.upsert(m["text"], m.get("type", "episodic"))
        except json.JSONDecodeError:
            pass

    def compact(self):
        if len(self.history) <= WINDOW:
            return
        old, self.history = self.history[:-WINDOW], self.history[-WINDOW:]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        self.summary = chat([{"role": "user", "content":
            f"Update the running conversation summary.\nCurrent summary: {self.summary or '(none)'}\n"
            f"New turns:\n{transcript}\nReturn a concise summary (max 5 sentences)."}])
        print("   [compact] older turns summarised")

    def reply(self, user: str) -> str:
        memories = self.ltm.recall(user)
        mem_txt = "\n".join(f"- ({m['type']}) {m['text']}" for m in memories) or "(none)"
        system = ("You are a helpful work assistant with long-term memory. Use the recalled memories when "
                  "relevant and do not mention memories that are irrelevant.\n"
                  f"Recalled memories:\n{mem_txt}\n"
                  f"Conversation summary so far: {self.summary or '(none)'}")
        out = chat([{"role": "system", "content": system}] + self.history + [{"role": "user", "content": user}])
        self.history += [{"role": "user", "content": user}, {"role": "assistant", "content": out}]
        self.extract(user, out)
        self.compact()
        return out


def main():
    global CHAT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--chat", action="store_true")
    a = ap.parse_args()
    CHAT_MODEL = a.model
    bot = MemoryChatbot()
    try:
        if a.chat:
            print("Memory chatbot. Commands: /memories /forget <id> /reset /new /exit")
            while (u := input("\nYou: ").strip()).lower() not in {"/exit", "exit", "quit"}:
                if u == "/memories":
                    for m in bot.ltm.items:
                        print(f"  {m['id']} [{m['type']}] {m['text']}")
                elif u.startswith("/forget"):
                    print("  deleted" if bot.ltm.forget(u.split()[-1]) else "  not found")
                elif u == "/reset":
                    bot.ltm.items = []; bot.ltm.save(); print("  store wiped")
                elif u == "/new":
                    bot.new_session(); print("  new session (long-term memory kept)")
                elif u:
                    print("Bot:", bot.reply(u))
            return
        # Scripted demo: session 1 teaches facts, session 2 relies on recall only
        print("── Session 1 ──")
        for u in ["Hi, I'm a GenAI engineer on the procurement automation team.",
                  "I prefer answers with Python code examples, and we use Ollama locally.",
                  "This week I'm building a KAG prototype for vendor eligibility."]:
            print(f"\nYou: {u}\nBot: {bot.reply(u)}")
        bot.new_session()
        print("\n── Session 2 (new conversation, short-term memory cleared) ──")
        for u in ["What was I working on this week?", "Suggest how I should test it."]:
            print(f"\nYou: {u}\nBot: {bot.reply(u)}")
        print(f"\nStored memories are in {STORE.name}. Run with --chat and use /memories to inspect them.")
    except (ConnectionError, ollama.ResponseError) as e:
        sys.exit(f"Ollama error: {e}")


if __name__ == "__main__":
    main()
